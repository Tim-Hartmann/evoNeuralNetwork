package neuralnet

import (
	"sort"
	"sync"
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
	var wg sync.WaitGroup
	wg.Add(ParallelValue)
	for i:=0; i < ParallelValue; i++ {
		go func(i int, input []float64){
			defer wg.Done()
			for c := i; c < len(n.Networks); c+= ParallelValue {
				n.Networks[c].Forward(input)
			}
		}(i, input)
	}
	wg.Wait()
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

	var wg sync.WaitGroup
	var networksLock sync.Mutex
	wg.Add(MaxRepop-SurvivorCount)
	for w := 0; w < ParallelValue; w++ {
		for i := SurvivorCount+w; i < MaxRepop; i+=ParallelValue {
			go func(i int) {
				defer wg.Done()
				net := n.Networks[i%SurvivorCount].Copy()
				net.Mutate()
				networksLock.Lock()
				newNet.Networks = append(newNet.Networks, net)
				networksLock.Unlock()
			}(i)
		}
	}
	wg.Wait()


	for i := 0; i < NewSpawnCount; i++ {
		newNet.Networks = append(newNet.Networks, RandomNet(n.Structure))
	}
	n.Networks = newNet.Networks
}
