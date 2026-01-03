# def get_curriculum_data(epoch):
#     curr_partition_upper = partition_size = max_iters // len(curriculum_parts)
#     for i, part in enumerate(curriculum_parts):
#         if epoch < curr_partition_upper:
#             return part
#
#         curr_partition_upper += partition_size
