module a11_softmax #(
    parameter K = 30,          // Number of inputs (fixed to 30)
    parameter IN_WIDTH = 11,   // Input width
    parameter OUT_WIDTH = 8    // Output width
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          valid_in,
    input  logic [K-1:0][IN_WIDTH-1:0]    data_in,  // 30x11b input
    output logic                          valid_out,
    output logic [K-1:0][OUT_WIDTH-1:0]   data_out  // 30x8b output
);

    // State machine definition
    typedef enum logic [2:0] {
        IDLE,
        FIND_MAX,        // Find max for numerical stability
        SUBTRACT_MAX,    // Subtract max from all inputs
        COMPUTE_EXP,     // Compute exponentials
        COMPUTE_SUM,     // Sum up exponentials
        NORMALIZE,       // Divide by sum
        DONE
    } state_t;

    state_t state;

    // Internal registers for computation
    logic [IN_WIDTH-1:0] max_val;
    logic [IN_WIDTH-1:0] adjusted_inputs [K-1:0];
    logic [15:0] exp_values [K-1:0];     // Wider to hold exp results
    logic [19:0] exp_sum;                // Wider for sum of 30 numbers
    logic [4:0] counter;                 // 5 bits to count up to 30

    // LUT for exponential approximation
    function logic [7:0] exp_lut(logic [3:0] x);
        case(x)
            4'h0: return 8'h01;  // exp(0) ≈ 1
            4'h1: return 8'h02;  // exp(1) ≈ 2.718
            4'h2: return 8'h07;  // exp(2) ≈ 7.389
            4'h3: return 8'h14;  // exp(3) ≈ 20.086
            4'h4: return 8'h37;  // exp(4) ≈ 54.598
            4'h5: return 8'h8E;  // exp(5) ≈ 148.413
            4'h6: return 8'hFF;  // exp(6) and above ≈ max value
            default: return 8'hFF;
        endcase
    endfunction

    // Initial assertion to verify K=30
    initial begin
        assert (K == 30) else $error("This module is designed for K=30 only");
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid_out <= 1'b0;
            max_val <= '0;
            counter <= '0;
            exp_sum <= '0;
            for (int i = 0; i < K; i++) begin
                adjusted_inputs[i] <= '0;
                exp_values[i] <= '0;
                data_out[i] <= '0;
            end
        end else begin
            case (state)
                IDLE: begin
                    if (valid_in) begin
                        state <= FIND_MAX;
                        max_val <= data_in[0];
                        counter <= 1;
                        valid_out <= 1'b0;
                    end
                end

                FIND_MAX: begin
                    // Find maximum value for numerical stability
                    if (data_in[counter] > max_val) begin
                        max_val <= data_in[counter];
                    end
                    
                    if (counter == K-1) begin
                        state <= SUBTRACT_MAX;
                        counter <= '0;
                    end else begin
                        counter <= counter + 1;
                    end
                end

                SUBTRACT_MAX: begin
                    // Subtract max from all inputs to prevent overflow
                    if (counter < K) begin
                        adjusted_inputs[counter] <= data_in[counter] - max_val;
                        counter <= counter + 1;
                    end else begin
                        state <= COMPUTE_EXP;
                        counter <= '0;
                        exp_sum <= '0;
                    end
                end

                COMPUTE_EXP: begin
                    // Compute exponentials using LUT
                    if (counter < K) begin
                        // Scale down to 4 bits for LUT
                        exp_values[counter] <= exp_lut(adjusted_inputs[counter][IN_WIDTH-1:IN_WIDTH-4]);
                        counter <= counter + 1;
                    end else begin
                        state <= COMPUTE_SUM;
                        counter <= '0;
                    end
                end

                COMPUTE_SUM: begin
                    // Sum up all exponentials
                    if (counter < K) begin
                        exp_sum <= exp_sum + exp_values[counter];
                        counter <= counter + 1;
                    end else begin
                        state <= NORMALIZE;
                        counter <= '0;
                    end
                end

                NORMALIZE: begin
                    // Normalize by dividing each exp value by sum
                    if (counter < K) begin
                        // Scale to OUT_WIDTH bits
                        data_out[counter] <= (exp_values[counter] << OUT_WIDTH) / exp_sum;
                        counter <= counter + 1;
                    end else begin
                        state <= DONE;
                    end
                end

                DONE: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule