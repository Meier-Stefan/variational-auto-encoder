import torch


def compare_models(file1, file2):
    """
    Compare two PyTorch model files.

    Args:
        file1: Path to the first .pth file.
        file2: Path to the second .pth file.

    Returns:
        dict: {"identical": bool, "differences": list of str}
    """
    state1 = torch.load(file1, map_location="cpu")
    state2 = torch.load(file2, map_location="cpu")

    differences = []

    for key in state1:
        if key not in state2:
            differences.append(f"Missing key in second model: {key}")
            continue

        if not torch.equal(state1[key], state2[key]):
            differences.append(f"Mismatch in parameter: {key}")

    identical = len(differences) == 0

    return {"identical": identical, "differences": differences}


if __name__ == "__main__":
    file1 = "vae_mnist.pth"
    file2 = "vae_mnist1.pth"

    result = compare_models(file1, file2)

    if result["identical"]:
        print("Models are IDENTICAL")
    else:
        print("Models differ")
        for diff in result["differences"]:
            print(diff)