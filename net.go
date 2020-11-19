package neuralnet

import (
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"reflect"
	"runtime"
)

type Net struct {
	ID      string
	Species string
	Nodes   [][]Node
	Output  []float64
	Error   float64
}

type Node struct {
	acc        float64
	activation func(float64) float64
	weights    []float64 //weights for passing onto the next layer
}

func ReLU(x float64) float64 {
	var f func(float64) float64
	f = Sigmoid
	f(3)
	if x < 0 {
		return 0
	} else {
		return x
	}
}
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}
func Binary(x float64) float64 {
	if x < 0.5 {
		return 0
	} else {
		return 1
	}
}
func Identity(x float64) float64 {
	return x
}
func Square(x float64) float64 {
	return math.Pow(x, 2)
}
func AbsRoot(x float64) float64 {
	return math.Sqrt(math.Abs(x))
}

var Activations = []func(float64) float64{
	Sigmoid,
	ReLU,
	Binary,
	Identity,
	Square,
	AbsRoot,
	math.Sin,
	math.Cos,
	math.Ceil,
	math.Floor,
}

var letterRunes = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

func randomString(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letterRunes[rand.Intn(len(letterRunes))]
	}
	return string(b)
}

func randomActivation() func(float64) float64 {
	return Activations[rand.Intn(len(Activations))]
}

func randomNode(nextLayerWidth int, rndSource *rand.Rand) Node {
	n := Node{
		acc:        0,
		activation: randomActivation(),
		weights:    []float64{},
	}
	for i := 0; i < nextLayerWidth; i++ {
		n.weights = append(n.weights, rndSource.Float64()*2.0-1.0)
	}
	return n
}
func (n Node) mutate(rndSource *rand.Rand) Node {
	if rndSource.Intn(400) == 0 { // Replace activation function and reseed weights
		n.activation = randomActivation()
		for i := 0; i < len(n.weights); i++ {
			n.weights[i] = rndSource.Float64()*2.0 - 1
		}
	} else { // Mutate the weights randomly
		for i := 0; i < len(n.weights); i++ {
			mutate(&n.weights[i], rndSource)
		}
	}
	return n
}
func mutate(w *float64, rndSource *rand.Rand) {
	if rand.Intn(3) == 0 {
		*w += (rndSource.Float64() - 0.5) / 15.0
	} else if rndSource.Intn(100) == 1 {
		*w = rndSource.Float64()*2.0 - 1
	}
}

func RandomNet(layers []int, rndSource *rand.Rand) Net {
	n := Net{
		ID:      randomString(10),
		Species: randomString(5),
		Error:   0.0,
	}
	// Create output slots
	for i := 0; i < layers[len(layers)-1]; i++ {
		n.Output = append(n.Output, 0.0)
	}

	// Create Nodes
	for l := 0; l < len(layers)-1; l++ {
		var currentLayerNodes []Node
		for n := 0; n < layers[l]; n++ {
			currentLayerNodes = append(currentLayerNodes, randomNode(layers[l+1], rndSource))
		}
		n.Nodes = append(n.Nodes, currentLayerNodes)
	}

	return n
}

func (net *Net) Forward(input []float64) {
	if len(input) != len(net.Nodes[0]) {
		log.Fatal("Input data must match length of first layer of network")
	}
	//Reset net output layer
	for i := 0; i < len(net.Output); i++ {
		net.Output[i] = 0.0
	}

	// Reset all nodes
	for i := 0; i < len(net.Nodes); i++ {
		for j := 0; j < len(net.Nodes[i]); j++ {
			net.Nodes[i][j].acc = 0.0
		}
	}

	//Feed input into first layer of network
	for i := 0; i < len(net.Nodes[0]); i++ {
		net.Nodes[0][i].acc = input[i]
	}
	for l := 0; l < len(net.Nodes)-1; l++ { // Forward all layers except last one
		for n := 0; n < len(net.Nodes[l]); n++ {
			// apply activation to acc
			net.Nodes[l][n].acc = net.Nodes[l][n].activation(net.Nodes[l][n].acc)
			// multiply acc by weight for each node of the next layer and add it to its acc
			for i, w := range net.Nodes[l][n].weights {
				net.Nodes[l+1][i].acc += net.Nodes[l][n].acc * w
			}
		}
	}
	// Forward the last layer into the output layer
	for _, n := range net.Nodes[len(net.Nodes)-1] {
		n.acc = n.activation(n.acc)
		for i, w := range n.weights {
			net.Output[i] += n.acc * w
		}
	}
}

func (net *Net) Mutate(rndSource *rand.Rand) {
	net.ID = randomString(10)
	net.Error = 0.0
	for i := 0; i < len(net.Nodes); i++ {
		for j := 0; j < len(net.Nodes[i]); j++ {
			net.Nodes[i][j].mutate(rndSource)
		}
	}
}

func (net *Net) Copy() Net {
	result := Net{
		ID:      net.ID,
		Species: net.Species,
		Nodes:   [][]Node{},
		Output:  []float64{},
		Error:   0,
	}
	// Create output fields
	for _, _ = range net.Output {
		result.Output = append(result.Output, 0.0)
	}

	//Add all nodes
	for _, layer := range net.Nodes {
		newLayer := []Node{}
		for _, node := range layer {
			newNode := Node{
				acc:        0,
				activation: node.activation,
				weights:    []float64{},
			}
			for _, w := range node.weights {
				newNode.weights = append(newNode.weights, w)
			}
			newLayer = append(newLayer, newNode)
		}
		result.Nodes = append(result.Nodes, newLayer)
	}
	return result
}

func getNameOfFunction(i interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
}

func PrettyPrintNet(net [][]Node) {
	type textNode struct {
		Weights    []float64
		Activation string
	}
	res := [][]textNode{}

	for j := 0; j < len(net); j++ {
		layer := []textNode{}
		for i := 0; i < len(net[j]); i++ {
			layer = append(layer, textNode{
				Activation: getNameOfFunction(net[j][i].activation),
				Weights:    net[j][i].weights,
			})
		}
		res = append(res, layer)
	}

	v, _ := json.Marshal(res)
	println(string(v))
}

func CheckNetworkError(error *float64) {
	if math.IsNaN(*error) {
		*error = math.Inf(1)
	}
}
