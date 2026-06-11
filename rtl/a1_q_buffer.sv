module a1_q_buffer (
    input logic clk,                // Clock input
    input logic rst_n,              // Active-low reset input
    input logic wr_en,              // Write enable input
    input logic [1023:0] data_in,   // 1024-bit input data
    output logic [1023:0] data_out  // 1024-bit output data
);

    // Internal signals
    logic [1023:0] temp_data;

    // Data storage and output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 0;
        end else begin
            if (wr_en) begin
                temp_data = data_in;
                data_out <= temp_data;
            end
        end
    end

endmodule