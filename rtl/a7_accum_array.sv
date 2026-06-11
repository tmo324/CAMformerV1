module a7_accum_array (
    input logic clk,                    // Clock input
    input logic rst_n,                  // Active-low reset input
    input logic valid_in,               // Input data valid signal
    input logic [31:0][10:0] data_in,   // 32 elements, 11 bits each
    input logic clear_acc,              // Clear accumulator signal
    output logic [31:0][10:0] data_out, // 32 elements, 11 bits each
    output logic valid_out              // Output data valid signal
);

    // Internal signals
    logic [31:0][10:0] temp_data;
    logic [31:0][10:0] accum_reg;

    // Accumulator logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_reg <= '0;
            data_out <= '0;
            valid_out <= 1'b0;
        end else begin
            if (clear_acc) begin
                accum_reg <= '0;
                valid_out <= 1'b0;
            end else if (valid_in) begin
                // Perform accumulation for each element
                for (int i = 0; i < 32; i++) begin
                    temp_data[i] = accum_reg[i] + data_in[i];
                    // Handle overflow by saturating at max value (2^11 - 1)
                    if (temp_data[i] > 11'h7FF) begin
                        accum_reg[i] <= 11'h7FF;
                    end else begin
                        accum_reg[i] <= temp_data[i];
                    end
                end
                data_out <= accum_reg;
                valid_out <= 1'b1;
            end else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule