import c3d

# with open('data_files/AIM_model_movements0004.c3d', 'rb') as handle:
#     reader = c3d.Reader(handle)
#     for i, (points, analog) in enumerate(reader.read_frames()):
#         print('Frame {}: {}'.format(i, points.round(2)))

print(c3d.__dir__())