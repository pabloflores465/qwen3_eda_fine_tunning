`timescale 1ns/1ps

module tb_counter4bit;

    reg clk;
    reg rst;
    wire [3:0] count;

    // Instantiate the DUT (Device Under Test)
    counter4bit dut (
        .clk(clk),
        .rst(rst),
        .count(count)
    );

    // Clock generation: 10 ns period (100 MHz equivalent)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // toggle every 5 ns
    end

    // Stimulus
    initial begin
        // Initialize
        rst = 1;
        #20;           // keep reset high for 20 ns

        rst = 0;       // release reset
        #200;          // let the counter run for a while

        rst = 1;       // assert reset again
        #20;
        rst = 0;       // release reset
        #100;

        $finish;       // end simulation
    end

    // Monitor to see values in the console
    initial begin
        $display("Time   rst  count");
        $monitor("%4t   %b    %b", $time, rst, count);
    end

    // Optional: dump waves for GTKWave or other viewers
    initial begin
        $dumpfile("counter4bit.vcd");
        $dumpvars(0, tb_counter4bit);
    end

endmodule

