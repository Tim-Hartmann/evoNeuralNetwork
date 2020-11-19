package neuralnet

import (
	"sort"
	"sync"
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
	for i := 0; i < MaxRepop; i++ {
		n.Networks = append(n.Networks, RandomNet(n.Structure))
	}
}

func (n *NetPool) Forward(input []float64) { //Forward all networks in parallel, do not calculate error
	var wg sync.WaitGroup
	for i := 0; i < len(n.Networks); i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			n.Networks[i].Forward(input)
		}(i)
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

	var networksLock sync.Mutex
	var wg sync.WaitGroup

	for i := SurvivorCount; i < MaxRepop; i++ {
		wg.Add(1)
		go func(i int){
			defer wg.Done()
			net := n.Networks[i%SurvivorCount].Copy()
			net.Mutate()

			networksLock.Lock()
			newNet.Networks = append(newNet.Networks, net)
			networksLock.Unlock()
		}(i)
	}
	wg.Wait()

	for i := 0; i < NewSpawnCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			net := RandomNet(n.Structure)

			networksLock.Lock()
			newNet.Networks = append(newNet.Networks, net)
			networksLock.Unlock()
		}()

	}
	n.Networks = newNet.Networks
}
