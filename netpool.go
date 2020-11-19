package neuralnet

import (
	"math/rand"
	"sort"
	"time"
)

var (
	MaxRepop = 25
	SurvivorCount = 5
	NewSpawnCount = 5
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
	rndSource := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	for i := 0; i < MaxRepop; i++ {
		n.Networks = append(n.Networks, RandomNet(n.Structure, rndSource))
	}
}

func (n *NetPool) Forward(input []float64) { //Forward all networks in parallel, do not calculate error
	for c := 0; c < len(n.Networks); c++ {
		n.Networks[c].Forward(input)
	}
}


func (n *NetPool) ResetErrors() {
	for i := 0; i < len(n.Networks); i++ {
		n.Networks[i].Error = 0.0
	}
}

func (n *NetPool) EvolutionStep() { // Make sure to set error manually before calling this function
	rndSource := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))

	sort.Slice(n.Networks, func(i, j int) bool {
		return n.Networks[i].Error < n.Networks[j].Error
	})

	newNet := NetPool{}
	for i := 0; i < SurvivorCount; i++ {
		newNet.Networks = append(newNet.Networks, n.Networks[i].Copy())
	}

	for i := SurvivorCount; i < MaxRepop; i++ {
		net := n.Networks[i%SurvivorCount].Copy()
		net.Mutate(rndSource)
		newNet.Networks = append(newNet.Networks, net)
	}

	for i := 0; i < NewSpawnCount; i++ {
		newNet.Networks = append(newNet.Networks, RandomNet(n.Structure, rndSource))
	}
	n.Networks = newNet.Networks
}
