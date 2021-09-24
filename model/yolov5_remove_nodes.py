import onnx
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Remove Slice and concat from YOLOV5")
    parser.add_argument('input_model', type=str)
    parser.add_argument('-o', dest='output_model', type=str, required=True)
    args = parser.parse_args()

    input_model = args.input_model
    output_path = args.output_model

    print("input model: {}, output model: {}".format(input_model, output_path))
    onnx_model = onnx.load(input_model)
    graph = onnx_model.graph

    node_count = len(graph.node)
    to_remove = []
    for i in range(node_count):
        node = graph.node[i]
        if node.op_type != "Conv":
            to_remove.append(node)
        else:
            break

    for node in to_remove:
        graph.node.remove(node)

    graph.node[0].input[0] = graph.input[0].name

    old_input = graph.input[0]
    dim = old_input.type.tensor_type.shape.dim
    dim[1].dim_value *= 4
    dim[2].dim_value //= 2
    dim[3].dim_value //= 2

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path)



