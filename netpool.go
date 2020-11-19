package neuralnet

import (
	"sort"
)

var (
	MaxRepop = 25
	SurvivorCount = 5
	NewSpawnCount = 5
	ParallelValue = 8
)

type NetPool struct {
	Networks  []Net
	Structure []int
}

func (n NetPool) Len() int {
	return len(n.Networks)
}
func (n NetPool) Swap(i, j int) {
	n.Networks[i], n.Networks[j] = n.Networks[j], n.Networks[i]
}

func (n *NetPool) Seed() {
	for i := 0; i < MaxRepop; i++ {
		n.Networks = append(n.Networks, RandomNet(n.Structure))
	}
}

func (n *NetPool) Forward(input []float64) { //Forward all networks in parallel, do not calculate error
	for i:=0; i < ParallelValue; i++ {
		go func(i int){
			for c := i; c < len(n.Networks); c+= ParallelValue {
				n.Networks[c].Forward(input)
			}
		}(i)
	}
}

func (n *NetPool) ResetErrors() {
	for i := 0; i < len(n.Networks); i++ {
		n.Networks[i].Error = 0.0
	}
}

func (n *NetPool) EvolutionStep() { // Make sure to set error manually before calling this function
	sort.Slice(n.Networks, func(i, j int) bool {
		return n.Networks[i].Error < n.Networks[j].Error
	})

	newNet := NetPool{}
	for i := 0; i < SurvivorCount; i++ {
		newNet.Networks = append(newNet.Networks, n.Networks[i].Copy())

	}
	for i := SurvivorCount; i < MaxRepop; i++ {
		net := n.Networks[i%SurvivorCount].Copy()
		net.Mutate()
		newNet.Networks = append(newNet.Networks, net)
	}
	for i := 0; i < NewSpawnCount; i++ {
		newNet.Networks = append(newNet.Networks, RandomNet(n.Structure))
	}
	n.Networks = newNet.Networks
}
