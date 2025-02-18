# import torch
#
# # Define the tensor
# a = torch.tensor([[6, 1], [6, 1], [8, 0], [7, 1]])
#
# # Create new tensor to apply the transformations
# transformed = torch.empty_like(a)
#
# # Rule for the first element
# transformed[:, 0] = torch.where(a[:, 0] == 6, 0,
#                     torch.where(a[:, 0] == 8, 1,
#                     torch.where(a[:, 0] == 16, 2, -1)))  # Use -1 for unexpected values
#
# # Rule for the second element
# transformed[:, 1] = torch.where(a[:, 1] == 0, 0,
#                     torch.where(a[:, 1] == 1, 20, -1))  # Use -1 for unexpected values
#
# print(transformed)


# i want to make a script to transform node feat vectors for the better clustering
# sourcing tranform_table, this snippet write a script like above


def main():
    for i in range(7):
        read_go = 0
        transform_dict = {}
        # ----------------
        # get the numbers
        # ----------------
        with open("./transform_table") as f:
            for lines in f.readlines():
                if read_go == 0:
                    if len(lines) == 2 and lines[0] == str(i):
                        read_go = 1
                    else:
                        pass
                    continue
                else:
                    if len(lines) == 0:
                        break
                    else:
                        if len(lines) < 3:
                            break
                        else:
                            ele = [x.strip() for x in lines.split(",")]
                            original_val = ele[0]
                            new_val = ele[2].strip()
                            transform_dict[original_val] = new_val
        # -----------------
        # write a snippet
        # -----------------
        code_str = f"transformed[:, {i}] = "
        for keys, vals in transform_dict.items():
            code_str += f" torch.where(a[:, {i}] == {keys}, {vals},"
        parenthesis = ")" * len(transform_dict)
        code_str += " -2"
        code_str += parenthesis
        print(code_str)



if __name__ == '__main__':
    main()