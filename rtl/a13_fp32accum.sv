module a13_fp32accum #(
    parameter SIZE = 1024,
    parameter BF16_WIDTH = 16,
    parameter FP32_WIDTH = 32
)(
    input  logic                            clk,
    input  logic                            rst_n,
    input  logic                            start,
    input  logic [SIZE-1:0][BF16_WIDTH-1:0] data_in,    // 1024 BF16 values
    input  logic                            valid_in,
    output logic [SIZE-1:0][BF16_WIDTH-1:0] data_out,   // 1024 BF16 results
    output logic                            valid_out,
    output logic                            done
);

    // State machine definition
    typedef enum logic [2:0] {
        IDLE = 3'b000,
        CONVERT = 3'b001,
        ACCUMULATE = 3'b010,
        ROUND = 3'b011,
        OUTPUT = 3'b100
    } state_t;
    
    state_t state, next_state;

    // Registers for pipeline stages
    logic [SIZE-1:0][FP32_WIDTH-1:0] fp32_values;     // Converted FP32 values
    logic [SIZE-1:0][FP32_WIDTH-1:0] accum_reg;       // Accumulator registers
    logic [9:0] counter;                              // Counter for accumulation
    
    // Convert BF16 to FP32
    function automatic logic [FP32_WIDTH-1:0] bf16_to_fp32(logic [BF16_WIDTH-1:0] bf16_in);
        logic [FP32_WIDTH-1:0] fp32_out;
        fp32_out = {bf16_in[15:0], 16'b0}; // Extend mantissa with zeros
        return fp32_out;
    endfunction
    
    // Convert FP32 to BF16 with rounding
    function automatic logic [BF16_WIDTH-1:0] fp32_to_bf16(logic [FP32_WIDTH-1:0] fp32_in);
        logic [BF16_WIDTH-1:0] bf16_out;
        logic round_bit;
        
        round_bit = fp32_in[16]; // Get rounding bit
        
        // Basic rounding to nearest even
        if (round_bit && (fp32_in[15:0] != '0 || fp32_in[17])) begin
            bf16_out = fp32_in[31:16] + 1'b1;
        end else begin
            bf16_out = fp32_in[31:16];
        end
        
        return bf16_out;
    endfunction

    // Main state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            counter <= '0;
            valid_out <= 1'b0;
            done <= 1'b0;
            for (int i = 0; i < SIZE; i++) begin
                fp32_values[i] <= '0;
                accum_reg[i] <= '0;
                data_out[i] <= '0;
            end
        end else begin
            state <= next_state;
            valid_out <= 1'b0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        counter <= '0;
                        done <= 1'b0;
                        // Clear accumulators
                        for (int i = 0; i < SIZE; i++) begin
                            accum_reg[i] <= '0;
                        end
                    end
                end

                CONVERT: begin
                    if (valid_in) begin
                        // Convert input BF16 to FP32
                        for (int i = 0; i < SIZE; i++) begin
                            fp32_values[i] <= bf16_to_fp32(data_in[i]);
                        end
                    end
                end

                ACCUMULATE: begin
                    // Accumulate in FP32
                    for (int i = 0; i < SIZE; i++) begin
                        accum_reg[i] <= accum_reg[i] + fp32_values[i];
                    end
                    counter <= counter + 1;
                end

                ROUND: begin
                    // Convert accumulated FP32 values back to BF16
                    for (int i = 0; i < SIZE; i++) begin
                        data_out[i] <= fp32_to_bf16(accum_reg[i]);
                    end
                    valid_out <= 1'b1;
                end

                OUTPUT: begin
                    done <= 1'b1;
                end
            endcase
        end
    end

    // Next state logic
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: 
                if (start) next_state = CONVERT;
            
            CONVERT:
                if (valid_in) next_state = ACCUMULATE;
            
            ACCUMULATE:
                if (counter == 10'h3FF) next_state = ROUND;  // After accumulating all values
            
            ROUND:
                next_state = OUTPUT;
            
            OUTPUT:
                next_state = IDLE;
            
            default:
                next_state = IDLE;
        endcase
    end

endmodule