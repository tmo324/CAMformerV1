module a13_denseMM (
    input  logic                clk,
    input  logic                rst_n,
    input  logic [239:0]        prob_in,     // 30x8b
    input  logic [15359:0]      value_in,    // 30x32xBF16
    output logic [511:0]        result       // 32x1xBF16
);
    // For each output position
    logic [4:0]   pos_counter;    // 0-31
    logic [4:0]   mul_counter;    // 0-29
    logic [15:0]  mul_result;
    logic [15:0]  sum_reg;

    // Control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pos_counter <= 5'h0;
            mul_counter <= 5'h0;
        end else begin
            if (mul_counter == 5'd29) begin
                mul_counter <= 5'h0;
                if (pos_counter == 5'd31) begin
                    pos_counter <= 5'h0;
                end else begin
                    pos_counter <= pos_counter + 1;
                end
            end else begin
                mul_counter <= mul_counter + 1;
            end
        end
    end

    // Multiply and accumulate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mul_result <= 16'h0;
            sum_reg <= 16'h0;
        end else begin
            mul_result <= prob_in[mul_counter*8 +: 8] * 
                         value_in[(pos_counter*30 + mul_counter)*16 +: 16];
            
            if (mul_counter == 5'h0) begin
                sum_reg <= mul_result;
            end else begin
                sum_reg <= sum_reg + mul_result;
            end
        end
    end

    // Store result
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= '0;
        end else if (mul_counter == 5'd29) begin
            result[pos_counter*16 +: 16] <= sum_reg + mul_result;
        end
    end

endmodule