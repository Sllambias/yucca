from batchgenerators.utilities.file_and_folder_operations import load_json


class node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.channel = None
        self.layer = None


class LossTree:
    def __init__(self, TreeDict):
        if isinstance(TreeDict, str):
            TreeDict = load_json(TreeDict)
        self.root = node("Tree Root")
        self.layers = []
        self.loss_layers = []
        self.recursive_expand(TreeDict)
        self.layers = set(self.layers)
        self.get_loss_layers()

    def recursive_expand(self, treedict, starting_node=None, starting_layer=-1):
        starting_layer += 1
        if not starting_node:
            starting_node = self.root
        for key, value in treedict.items():
            newnode = node(key)
            newnode.layer = starting_layer
            self.layers.append(newnode.layer)
            starting_node.children.append(newnode)
            if not isinstance(value, dict):
                newnode.channel = value
            else:
                self.recursive_expand(value, starting_node=newnode, starting_layer=starting_layer)

    def get_loss_layers(self):
        for layer in self.layers:
            layer_list = []
            self.loss_layers.append(self.recursive_loss_layers(layer, self.root, layer_list))

    def get_subclasses(self, node, class_list):
        if len(node.children) == 0:
            class_list.append(node.channel)
        else:
            for child in node.children:
                self.get_subclasses(child, class_list)
        return class_list

    def print(self, node=None, indent=0):
        if not node:
            node = self.root
            prefix = ""
        else:
            prefix = "|- " + str(node.layer) + ". "

        if len(node.children) == 0:
            print(" " * indent + prefix + node.name + " = " + str(node.channel))
        else:
            print(" " * indent + prefix + node.name)
            for child in node.children:
                self.print(child, indent + 4)

    def recursive_loss_layers(self, layer, starting_node, layer_list):
        for child in starting_node.children:
            if len(child.children) == 0 or child.layer == layer:
                layer_list.append(self.get_subclasses(child, []))
            else:
                self.recursive_loss_layers(layer, child, layer_list=layer_list)
        return layer_list
