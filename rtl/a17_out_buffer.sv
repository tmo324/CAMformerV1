module a17_out_buffer (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        wr_en,
    input  logic [9:0]  addr,    // 10 bits for 1024 depth
    input  logic [15:0] wr_data,
    output logic [15:0] rd_data
);

    logic [15:0] mem [0:1023];  // 1024 x 16-bit memory array

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 1024; i++) begin
                mem[i] <= '0;
            end
        end
        else if (wr_en) begin
            mem[addr] <= wr_data;
        end
    end

    assign rd_data = mem[addr];

endmodule