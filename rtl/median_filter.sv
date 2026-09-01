// median_filter.sv — Phase 16
//
// Combinational 3×3 median filter: the 19-comparator median-of-9 network
// (Paeth, "Median Finding on a 3x3 Grid", Graphics Gems I).
//
// WHY THIS NETWORK AND NOT A HAND-ROLLED ONE
// ------------------------------------------
// The previous implementation sorted p[0..3] and p[4..7] with two 4-sorters,
// applied a bitonic half-cleaner across them, and then read the median off
// t4[3] and t4[4]. That is wrong, and quietly so. A half-cleaner guarantees
// only that every element of the lower half is <= every element of the upper
// half; each half is left BITONIC, not sorted. So t4[3] is not max(lower) and
// t4[4] is not min(upper), which is what the tail assumed. Brute-forcing the
// old sequence over all 9! = 362880 orderings of nine distinct values returned
// a value other than the true median on 299520 of them — 82.54%. On a real
// salt-and-pepper image 72.6% of pixels were wrong, with a worst-case error of
// 225 grey levels, against the bit-exact budget of 0 in configs/hardware.yaml.
//
// The tail could not be patched: max(lower) and min(upper) were not present on
// any wire it read. The network below replaces it wholesale and is verified to
// zero mismatches over all 9! permutations, the exhaustive 4^9 duplicate space,
// and 200000 random uint8 windows. Do not "optimise" a comparator out of it
// without re-running that check — 19 is the smallest known correct network for
// the median of nine, so any removal makes it wrong.
//
// Each step is a compare-exchange: after CS(i,j) the SMALLER value is in v[i]
// and the larger in v[j]. Note that steps 17 and 19 are written (4,2) and step
// 18 is (6,4) — the descending index order is deliberate, not a typo. The
// minimum always lands in the FIRST index named.
//
// Latency: fully combinational (0 clocks). Register the output in the caller if
// timing requires; filter_controller.sv already does.

`default_nettype none

module median_filter #(
    parameter int DEPTH = 8
) (
    input  logic [3*3*DEPTH-1:0] win_flat,   // flat; element [r][c] = win_flat[(r*3+c)*DEPTH +: DEPTH]
    output logic [DEPTH-1:0] median_out
);
    // Flatten the 3×3 window into 9 wires using the flat port directly.
    // win_flat[(r*3+c)*DEPTH +: DEPTH] = element [r][c].
    logic [DEPTH-1:0] p [0:8];
    assign p[0] = win_flat[(0*3+0)*DEPTH +: DEPTH];
    assign p[1] = win_flat[(0*3+1)*DEPTH +: DEPTH];
    assign p[2] = win_flat[(0*3+2)*DEPTH +: DEPTH];
    assign p[3] = win_flat[(1*3+0)*DEPTH +: DEPTH];
    assign p[4] = win_flat[(1*3+1)*DEPTH +: DEPTH];
    assign p[5] = win_flat[(1*3+2)*DEPTH +: DEPTH];
    assign p[6] = win_flat[(2*3+0)*DEPTH +: DEPTH];
    assign p[7] = win_flat[(2*3+1)*DEPTH +: DEPTH];
    assign p[8] = win_flat[(2*3+2)*DEPTH +: DEPTH];

    // Working registers for the network. Blocking assignments inside
    // always_comb execute in written order, so this describes exactly the
    // 19-stage comparator cascade and synthesises to the same.
    logic [DEPTH-1:0] v [0:8];
    logic [DEPTH-1:0] t;

    always_comb begin
        for (int i = 0; i < 9; i++) v[i] = p[i];

        // 1..3   sort the three column triples' middles
        if (v[1] > v[2]) begin t = v[1]; v[1] = v[2]; v[2] = t; end
        if (v[4] > v[5]) begin t = v[4]; v[4] = v[5]; v[5] = t; end
        if (v[7] > v[8]) begin t = v[7]; v[7] = v[8]; v[8] = t; end
        // 4..6
        if (v[0] > v[1]) begin t = v[0]; v[0] = v[1]; v[1] = t; end
        if (v[3] > v[4]) begin t = v[3]; v[3] = v[4]; v[4] = t; end
        if (v[6] > v[7]) begin t = v[6]; v[6] = v[7]; v[7] = t; end
        // 7..9   each triple is now sorted
        if (v[1] > v[2]) begin t = v[1]; v[1] = v[2]; v[2] = t; end
        if (v[4] > v[5]) begin t = v[4]; v[4] = v[5]; v[5] = t; end
        if (v[7] > v[8]) begin t = v[7]; v[7] = v[8]; v[8] = t; end
        // 10..12 discard the impossible extremes
        if (v[0] > v[3]) begin t = v[0]; v[0] = v[3]; v[3] = t; end
        if (v[5] > v[8]) begin t = v[5]; v[5] = v[8]; v[8] = t; end
        if (v[4] > v[7]) begin t = v[4]; v[4] = v[7]; v[7] = t; end
        // 13..15
        if (v[3] > v[6]) begin t = v[3]; v[3] = v[6]; v[6] = t; end
        if (v[1] > v[4]) begin t = v[1]; v[1] = v[4]; v[4] = t; end
        if (v[2] > v[5]) begin t = v[2]; v[2] = v[5]; v[5] = t; end
        // 16..19 converge on the 5th order statistic
        if (v[4] > v[7]) begin t = v[4]; v[4] = v[7]; v[7] = t; end
        if (v[4] > v[2]) begin t = v[4]; v[4] = v[2]; v[2] = t; end
        if (v[6] > v[4]) begin t = v[6]; v[6] = v[4]; v[4] = t; end
        if (v[4] > v[2]) begin t = v[4]; v[4] = v[2]; v[2] = t; end
    end

    // v[4] now holds the median of the nine input pixels.
    assign median_out = v[4];

endmodule

`default_nettype wire
