# Tensorflow Graph Inspection Tricks
Often you want to hook into an existing model. Here are some tricks for finding tensors that you care about

#### Find all the input tensors to `target_node`

```python
target_node = graph.get_tensor_by_name("predictions:0")

target_node.op.inputs
```

#### Find all the output tensors of `target_node`

```python
target_node = graph.get_tensor_by_name("predictions:0")

[output for op in target_node.consumers() for output in op.outputs]
```

#### Find all the tensors that have a certain shape
```python
target_shape = (None, 32, 32, 3)

def shape_tuple(tensor):
  return tuple(map(lambda i: i.value, tensor.get_shape()))
  
tensors = tf.contrib.graph_editor.get_tensors(graph)
[t for t in tensors if shape_tuple(t) == target_shape]
```

#### Find all the tensors with `target_name` as a substring
```python
target_name = 'logit'

tensors = tf.contrib.graph_editor.get_tensors(graph)
[t for t in tensors if target_name in t.name]
```
