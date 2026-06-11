module a5_x8 (
    input  logic        clk,      // Clock input
    input  logic        rst_n,    // Active-low reset
    input  logic [8:0]  data_in,  // 9-bit input
    output logic [11:0] data_out  // 12-bit output
);

    // Internal signal for multiplication
    logic [11:0] temp_result;

    // Sequential logic with async reset
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 12'b0;  // Clear output on reset
        end else begin
            temp_result = {data_in, 3'b000};  // Multiply by shifting left 3 positions
            data_out <= temp_result;          // Register the result
        end
    end

endmodule