module a6_sub32 #(
    parameter CAMH = 32  // Modified by script
)(
    input  logic        clk,          // Clock input
    input  logic        rst_n,        // Active-low reset
    input  logic [11:0] data_in,      // 12-bit input
    output logic [10:0] data_out      // 11-bit output
);

    // Internal signal for subtraction
    logic [11:0] temp_result;

    // Sequential logic with async reset
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 11'b0;  // Clear output on reset
        end else begin
            temp_result = data_in - CAMH;     // Perform subtraction
            data_out <= temp_result[10:0];    // Take lower 11 bits for output
        end
    end

endmodule
