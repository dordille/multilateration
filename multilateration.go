package main

import (
	"math"
)

type Multilateration struct {
	Positions [][]float64
	Distances []float64
}

func (f Multilateration) Func(point []float64) float64 {
	return f.relativeError(point)
}

func (f Multilateration) Grad(grad []float64, point []float64) {
	currentError := f.relativeError(point)
	dd := 1e-6

	for i := 0; i < len(grad); i++ {
		di := make([]float64, len(grad))
		di[i] = dd
		grad[i] = (f.relativeError(add(point, di)) - currentError) / dd
	}
}

func (f Multilateration) InitialPoint() []float64 {
	numberOfPositions := len(f.Positions)
	positionDimension := len(f.Positions[0])
	initialPoint := make([]float64, positionDimension)

	for i := 0; i < len(f.Positions); i++ {
		for j := 0; j < len(f.Positions[i]); j++ {
			initialPoint[j] += f.Positions[i][j]
		}
	}

	for j := 0; j < len(initialPoint); j++ {
		initialPoint[j] /= float64(numberOfPositions)
	}

	return initialPoint
}

func (f Multilateration) relativeError(point []float64) float64 {
	var sum float64
	for i := 0; i < len(f.Positions); i++ {
		distanceEstimate := magnitude(subtract(point, f.Positions[i]))
		sum += math.Pow(distanceEstimate-f.Distances[i], 2)
	}

	return sum / float64(len(f.Positions))
}

func magnitude(x []float64) float64 {
	var sqrs float64
	for i := 0; i < len(x); i++ {
		sqrs += math.Pow(x[i], 2)
	}

	return math.Sqrt(sqrs)
}

func add(x []float64, y []float64) []float64 {
	d := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		d[i] = x[i] + y[i]
	}

	return d
}

func subtract(x []float64, y []float64) []float64 {
	d := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		d[i] = x[i] - y[i]
	}

	return d
}
