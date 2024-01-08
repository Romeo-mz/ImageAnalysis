import numpy as np
def zhang_suen_thicken(binary, num_iter=1):
    output = np.pad(binary, 1, mode="constant", constant_values=0)
    ROW, COL = output.shape

    for _ in range(num_iter):
        # iteration 1
        added = np.zeros((ROW, COL), dtype=bool)
        for i in range(1, ROW - 1):
            for j in range(1, COL - 1):
                if output[i, j] == 0:
                    curr = output[i - 1 : i + 2, j - 1 : j + 2]
                    if rule1_thicken(curr) and rule2_thicken(curr) and (rule3_1(curr) or rule3_2(curr)):
                        added[i, j] = True

        # Apply addition of marked pixels
        output[1:-1, 1:-1][added[1:-1, 1:-1]] = 1  # Set marked pixels to 1 to thicken lines

    return output[1:-1, 1:-1]


def rule1_thicken(patch):
    num_neigh = 8 - np.sum(patch)  # Inverting the condition for thickening
    return 2 <= num_neigh <= 6


def rule2_thicken(patch):
    neighbors = [
        patch[0, 0], patch[0, 1], patch[0, 2],
        patch[1, 2], patch[2, 2], patch[2, 1],
        patch[2, 0], patch[1, 0], patch[0, 0]
    ]
    count = 0
    for i in range(8):
        if neighbors[i] == 1 and neighbors[i + 1] == 0:  # Adjusted condition for thickening
            count += 1
    return count == 1


def rule3_1(patch):
    trb = patch[0, 1] * patch[1, 2] * patch[2, 1]
    lbr = patch[1, 0] * patch[2, 1] * patch[1, 2]
    return (trb == 0) and (lbr == 0)


def rule3_2(patch):
    tlb = patch[0, 1] * patch[1, 0] * patch[2, 1]
    ltr = patch[1, 0] * patch[0, 1] * patch[1, 2]
    return (tlb == 0) and (ltr == 0)