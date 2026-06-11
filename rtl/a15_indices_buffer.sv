module a15_indices_buffer (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                wr_en,      // Write enable
    input  logic [30*8-1:0]     indices_in, // New indices to write
    output logic [30*8-1:0]     indices_out,// Buffered indices
    output logic                valid_out   // Indices valid indicator
);
    // Internal storage
    logic [30*8-1:0] buffer;
    logic            valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buffer    <= '0;
            valid    <= 1'b0;
        end else if (wr_en) begin
            buffer    <= indices_in;
            valid    <= 1'b1;
        end
    end

    // Continuous assignment for outputs
    assign indices_out = buffer;
    assign valid_out  = valid;
endmodule