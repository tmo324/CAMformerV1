`timescale 1ns/1ps

module tb_q_buffer;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic wr_en = 1'b0;
    logic [1023:0] data_in = '0;
    logic [1023:0] data_out;

    a1_q_buffer dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .data_in(data_in),
        .data_out(data_out)
    );

    always #5 clk = ~clk;

    initial begin
        repeat (2) @(posedge clk);
        #1;
        assert (data_out == '0) else $fatal(1, "reset did not clear data_out");

        @(negedge clk);
        rst_n = 1'b1;
        wr_en = 1'b1;
        data_in = {64{16'hA55A}};
        @(posedge clk);
        #1;
        assert (data_out == {64{16'hA55A}}) else $fatal(1, "write failed");

        @(negedge clk);
        wr_en = 1'b0;
        data_in = {64{16'h5AA5}};
        @(posedge clk);
        #1;
        assert (data_out == {64{16'hA55A}}) else $fatal(1, "hold failed");

        $display("PASS tb_q_buffer");
        $finish;
    end
endmodule
