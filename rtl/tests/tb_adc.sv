`timescale 1ns/1ps

module tb_adc;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic enable = 1'b0;
    logic [8:0] vin = '0;
    logic [8:0] dout;
    logic valid;

    a4_9bit_adc dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .vin(vin),
        .dout(dout),
        .valid(valid)
    );

    always #5 clk = ~clk;

    initial begin
        repeat (2) @(posedge clk);
        #1;
        assert (dout == '0 && !valid) else $fatal(1, "reset state is invalid");

        @(negedge clk);
        rst_n = 1'b1;
        enable = 1'b1;
        vin = 9'h12A;
        @(posedge clk);

        @(negedge clk);
        enable = 1'b0;
        @(posedge clk);

        @(negedge clk);
        vin = 9'h055;
        @(posedge clk);
        #1;
        assert (valid && dout == 9'h12A) else $fatal(1, "ADC conversion failed");

        @(posedge clk);
        #1;
        assert (!valid) else $fatal(1, "valid must be a one-cycle pulse");

        $display("PASS tb_adc");
        $finish;
    end
endmodule
