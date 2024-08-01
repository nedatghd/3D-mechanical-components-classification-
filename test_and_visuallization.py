from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import torch
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import pickle
result = {"predicted_label":[],"gt_label":[],"obj_path":[],"len_sample":int, "class_to_idx":{}}

def visuallize_mesh(path):
  mesh = o3d.io.read_triangle_mesh(path)
  if mesh.is_empty(): exit()

  triangles = np.asarray(mesh.triangles)
  vertices = np.asarray(mesh.vertices)
  colors = None
  if mesh.has_triangle_normals():
      colors = (0.5, 0.5, 0.5) + np.asarray(mesh.triangle_normals) * 0.5
      colors = tuple(map(tuple, colors))
  else:
      colors = (1.0, 0.0, 0.0)

  fig = go.Figure(
      data=[
          go.Mesh3d(
              x=vertices[:,0],
              y=vertices[:,1],
              z=vertices[:,2],
              i=triangles[:,0],
              j=triangles[:,1],
              k=triangles[:,2],
              facecolor=colors,
              opacity=0.50)
      ],
      layout=dict(
          scene=dict(
              xaxis=dict(visible=False),
              yaxis=dict(visible=False),
              zaxis=dict(visible=False)
          )
      )
  )
  return fig.show()

def find_keys_by_value(my_dict, value):
    keys = []
    for k, v in my_dict.items():
        if v == value:
            keys.append(k)
    return keys
    
def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        # print("---------------------------------")
        # print("test data", i, "is loaded ...")
        # print("len of data", len(data))
        # print("len of data", type(data["mesh"][0]))
        # print("true labels :", data["label"])
        model.set_input(data)
        with torch.no_grad():
            out = model.forward()
            # compute number of correct
            pred_class = out.data.max(1)[1]
            result["predicted_label"].append(pred_class.item())
            # print("pred_class", pred_class.item())
            result["obj_path"].append(dataset.dataset.__getitem__(i)["path"])
            # print(dataset.dataset.__getitem__(i)["path"])
            # print(dataset.dataset.classes)
            # print(dataset.dataset.class_to_idx)

            label_class = model.labels
            result["gt_label"].append(label_class.item())
            # print("label_class", label_class.item())
            # print("---------------------------------")
        result["len_sample"] = len(result["gt_label"])
        result["class_to_idx"] = dataset.dataset.class_to_idx
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return result





if __name__ == '__main__':
    result = run_test()
    # save dictionary to pred_label_dir.pkl file
    with open('pred_label_dir.pkl', 'wb') as fp:
        pickle.dump(result, fp)
        print('dictionary saved successfully pwd')

    
