module a9_convert (
    input  logic             clk,      // Clock input
    input  logic             rst_n,    // Active-low reset
    input  logic             valid_in, // Input valid signal
    input  logic [13:0]      data_in,  // 14-bit input
    output logic [10:0]      data_out, // 11-bit output
    output logic             valid_out  // Output valid signal
);

    // Internal signals
    logic [13:0] temp_data;
    
    // Conversion logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= '0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            
            if (valid_in) begin
                temp_data = data_in;
                // Saturate if value is larger than max 11-bit value
                if (data_in > 14'h7FF) begin
                    data_out <= 11'h7FF;  // Max value for 11 bits
                end else begin
                    data_out <= data_in[10:0];  // Take lower 11 bits
                end
            end
        end
    end

endmodule