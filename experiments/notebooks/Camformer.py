class Module:

    def __init__(self, power, area, delay, power2 = None, delay2 = None ):
        self.power = power # active power consumption of the module in mW
        self.area = area # area in um^2 
        self.delay = delay # delay of the module in cycles
        self.cycles = 0 # number of cycles executed by this module
        self.energy = 0
        self.power2 = power2 # optional secondary power consumption (for other modes of operation)
        self.delay2 = delay2 # optional secondary delay (for other modes of operation)

    def reset(self):
        self.cycles = 0
        self.energy = 0

    def run(self, times):

        self.cycles += self.delay * times
        self.energy += self.power * (self.delay * times) # energy in mW*cycles

    def run2(self, times):

        if self.power2 is None or self.delay2 is None:
            raise ValueError("Secondary power and delay not defined for this module.")
        
        self.cycles += self.delay2 * times
        self.energy += self.power2 * (self.delay2 * times)

    def __str__(self):
        return f"Module(power={self.power}, area={self.area}, delay={self.delay}, cycles={self.cycles}, energy={self.energy}, power2={self.power2}, delay2={self.delay2})"
        

class Camformer:
    def __init__(self, modules):
        """
        Initialize the Camformer with existing modules. (dependency injection)
        """
        try:
            self.query_buffer = modules["Query Buffer"]
            self.key_sram = modules["Key SRAM"]
            self.cap_cam_16x64 = modules["BA-CAM 16X64"]
            self.adcs = modules["6b ADCs"]
            self.fu_scalers = modules["Fixed Scalers"]
            self.top4 = modules["Top2"]  # using existing variable name for legacy code
            self.ptop_reg = modules["Ptop Reg"]
            self.top32 = modules["Top32"]
            self.softmax_lut = modules["SoftMax LUT"]
            self.softmax_divider = modules["SoftMax Div."]
            self.softmax_adder = modules["SoftMax Acc."]
            self.output_buffer = modules["Output Buff"]
            self.value_sram = modules["Value SRAM"]
            self.macs = modules["BF16 MACs"]
            self.dram = modules["DRAM"]
        except KeyError as e:
            raise ValueError(f"Missing required module: {e}. Please ensure all modules are provided in the initialization dictionary.")
        self.cycles = 0
        self.association_cycles = 0
        self.normalization_cycles = 0
        self.contextualization_cycles = 0

    def getEnergy(self):
        energy = 0
        energy += self.query_buffer.energy
        energy += self.key_sram.energy
        energy += self.cap_cam_16x64.energy
        energy += self.adcs.energy
        energy += self.fu_scalers.energy
        energy += self.top4.energy
        energy += self.ptop_reg.energy
        energy += self.top32.energy
        energy += self.softmax_lut.energy
        energy += self.softmax_divider.energy
        energy += self.softmax_adder.energy
        energy += self.output_buffer.energy
        energy += self.value_sram.energy
        energy += self.macs.energy
        #energy += self.dram.energy
        return energy

    def reset(self):
        self.cycles = 0
        self.association_cycles = 0
        self.normalization_cycles = 0
        self.contextualization_cycles = 0
        self.query_buffer.reset()
        self.key_sram.reset()
        self.cap_cam_16x64.reset()
        self.adcs.reset()
        self.fu_scalers.reset()
        self.top4.reset()
        self.ptop_reg.reset()
        self.top32.reset()
        self.softmax_lut.reset()
        self.softmax_divider.reset()
        self.softmax_adder.reset()
        self.output_buffer.reset()
        self.value_sram.reset()
        self.macs.reset()
        self.dram.reset()

    def _run_association(self, pipelined=True):
        """
        Run the association phase of the Camformer.
        """
        starting_cycles = self.cycles
        #Load the query buffer and key_sram
        self.query_buffer.run(1) 
        if not pipelined:
            for tile in range(1024//16):
                # program cap_cam 4 rows at a time
                for row_by_4 in range(16//4):
                    self.key_sram.run(1) # read key sram (4x64b = 256 wide access)
                    self.cap_cam_16x64.run2(1) # program
                    self.cycles += self.cap_cam_16x64.delay2 # programming delay
                # search 
                self.query_buffer.run(1) # read
                self.cap_cam_16x64.run(1) # search
                self.cycles += self.cap_cam_16x64.delay # search delay
                # adcs 
                self.adcs.run(1) # 6 bit adcs
                self.cycles += self.adcs.delay # add SAR delay
                # shifters and subs
                self.fu_scalers.run(1)
                self.cycles += self.fu_scalers.delay
                # top4
                self.top4.run(1)
                self.cycles += self.top4.delay # around 2 cycles
                # ptop_reg
                self.ptop_reg.run(1) # program top4 to ptop_reg
                self.cycles += self.ptop_reg.delay
                # top32 if necessary (all pipelined)
                if(tile >= 32 and tile % 16 == 0): # every 16 tiles, we need to find top32 of these 64 candidates
                    self.top32.run(1) # find top32 of these 64 candidates
                    self.cycles += self.top32.delay 
                    self.ptop_reg.run(1) # place 32 back in ptop_reg 
                    self.cycles += self.ptop_reg.delay
                # dram prefetch
                self.dram.run(1) # will read 128*2B
                self.value_sram.run(1) # will be stored in value sram
                # do not count dram/sram delay here as it is pipelined across stages
        else: # pipelined mode
            # We can pipeline fully except cam programming and search
            for tile in range(1024//16):
                # program cap_cam 4 rows at a time
                for row_by_4 in range(16//4):
                    self.key_sram.run(1) # read key sram (4x64 b wide access)
                    self.cap_cam_16x64.run2(1) # program
                    self.cycles += self.cap_cam_16x64.delay2 # programming delay
                # search 
                self.query_buffer.run(1) # read
                self.cap_cam_16x64.run(1) # search
                self.cycles += self.cap_cam_16x64.delay # search delay
                # adcs 
                self.adcs.run(1) # 6 bit adcs
                # rshifters
                self.fu_scalers.run(1)
                # top4
                self.top4.run(1)
                # ptop_reg
                self.ptop_reg.run(1)
                # top32 if necessary (all pipelined)
                if(tile >= 32 and tile % 16 == 0): # every 32 tiles, we need to find top32 of these 64 candidates
                    self.top32.run(1) # find top32 of these 64 candidates
                    self.ptop_reg.run(1) # place 32 back in ptop_reg 
                # dram prefetch
                self.dram.run(1) # one for each top4
                self.value_sram.run(1) # will be stored in value sram
                # do not count dram/sram delay here as it is pipelined across stages
            # after all tiles, account for draining the pipeline
            self.cycles += self.adcs.delay # add SAR delay
            self.cycles += self.fu_scalers.delay
            self.cycles += self.top4.delay # around 2 cycles
            self.cycles += self.ptop_reg.delay

        self.association_cycles += self.cycles - starting_cycles
        print(f"Total cycles for association phase: {self.cycles}")

    def _run_normalization(self, par_level, pipelined=True):
        """
        Run the normalization phase of the Camformer.
        """
        starting_cycles = self.cycles
        self.top32.run(1) # run top32 to get the (final) top 32 values from ptop_reg
        self.cycles += self.top32.delay # delay for top32 operation
        # run softmax loop
        for i in range(32 // par_level):
            self.softmax_lut.run(1)
            self.softmax_adder.run(1) # accumulate the softmax values
            self.output_buffer.run(1) # write 
            self.cycles += self.softmax_adder.delay # add delay for adder (parallelized)
        # run divider for softmax normalization
        for i in range(32 // par_level):
            self.output_buffer.run(1) # read 
            self.softmax_divider.run(1) # divide
            self.output_buffer.run(1) # write back normalized value
            if not pipelined:
                self.cycles += self.softmax_divider.delay
            else:
                self.cycles += 1
                
        if pipelined:
            self.cycles += self.softmax_divider.delay # if not pipelined, add delay for divider
        self.normalization_cycles += self.cycles - starting_cycles
        print(f"Total cycles for normalization phase: {self.cycles - starting_cycles}")

    def _run_contextualization(self, par_level, pipelined=True):
        """"
        Run the contextualization phase of the Camformer.
        """
        starting_cycles = self.cycles
        for mac in range(64*32 // par_level):
            """
            Run MACs for 16 values at a time (assuming 8 parallel MACs).
            """
            self.value_sram.run(1) # read value sram for the par_level values -- up to 16 (16x16b = 256b access)
            self.macs.run(1)
            if pipelined:
                self.cycles += 1 #macs pipelined
            else:
                self.cycles += self.macs.delay
        if pipelined:
            self.cycles += self.macs.delay # add delay for the last MAC operation 
        self.contextualization_cycles += self.cycles - starting_cycles
        print(f"Total cycles for contextualization phase: {self.cycles - starting_cycles}")

        





                               