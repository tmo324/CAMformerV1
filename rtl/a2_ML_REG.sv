module a2_ML_REG (
    input logic clk,                    // Clock input
    input logic rst_n,                  // Active-low reset input
    input logic wr_en,                  // Write enable input
    input logic [95:0][10:0] data_in,   // 96x11 bit input data
    output logic [95:0][10:0] data_out  // 96x11 bit output data
);

    // Internal signals
    logic [95:0][10:0] temp_data;

    // Data storage and output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= '0;  // Reset all elements to 0
        end else begin
            if (wr_en) begin
                temp_data = data_in;
                data_out <= temp_data;
            end
        end
    end

endmodule