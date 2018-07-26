package main

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/optimize"
)

type recorder struct {
}

func (recorder) Init() error {
	return nil
}

func (recorder) Record(l *optimize.Location, o optimize.Operation, s *optimize.Stats) error {
	if o == optimize.MajorIteration {
		fmt.Printf("Iteration [%d]: %v %v\n", s.MajorIterations, l.X, l.Gradient)
	}
	if o == optimize.GradEvaluation {
		fmt.Printf("\tX: %-20v Gradient: %-20v\n", l.X, l.Gradient)
	}
	if o == optimize.FuncEvaluation {
		fmt.Printf("\tX: %-20v Function: %-20v\n", l.X, l.F)
	}
	if o == optimize.PostIteration {
		fmt.Printf("\tX: %-20v\n", l.X)
	}
	return nil
}

func TestMagnitude(t *testing.T) {
	assert.Equal(t, magnitude([]float64{1.0}), 1.0)
	assert.Equal(t, magnitude([]float64{3.0, 4.0}), 5.0)
}

func TestMultilateration1DExact1(t *testing.T) {
	positions := [][]float64{{1.0}, {2.0}, {3.0}}
	distances := []float64{1.1, 0.1, 0.9}
	expected := []float64{2.1}

	result, err := multilaterationSolver(positions, distances)
	testResults(t, expected, 0.0001, err, result)
}

func TestMultilateration1DExact2(t *testing.T) {
	positions := [][]float64{{1000.0}, {2000.0}, {3000.0}}
	distances := []float64{1100, 100, 900}
	expected := []float64{2100.0}

	result, err := multilaterationSolver(positions, distances)
	testResults(t, expected, 0.0001, err, result)
}

func TestMultilateration1DInexact(t *testing.T) {
	positions := [][]float64{{1000.0}, {2000.0}, {3000.0}}
	distances := []float64{1110, 110, 910}
	expected := []float64{2100.0}

	result, err := multilaterationSolver(positions, distances)
	testResults(t, expected, 30, err, result)
}

func TestMultilateration2DExact1(t *testing.T) {
	positions := [][]float64{
		{1.0, 1.0},
		{3.0, 1.0},
		{2.0, 2.0},
	}
	distances := []float64{1.0, 1.0, 1.0}
	expected := []float64{2.0, 1.0}

	result, err := multilaterationSolver(positions, distances)
	testResults(t, expected, 0.001, err, result)
}

func TestMultilateration2DZeroDistance(t *testing.T) {
	positions := [][]float64{
		{1.0, 1.0},
		{2.0, 1.0},
	}
	distances := []float64{0.0, 1.0}
	expected := []float64{1.0, 1.0}

	result, err := multilaterationSolver(positions, distances)
	testResults(t, expected, 0.001, err, result)
}

func TestMultilateration2DExact2(t *testing.T) {
	positions := [][]float64{
		{0.0, 0.0},
		{-1.0, 0.0},
		{0.0, -1.0},
	}
	distances := []float64{math.Sqrt(2.0), 1.0, 1.0}
	expected := []float64{-1.0, -1.0}

	result, err := multilaterationSolver(positions, distances)
	testResults(t, expected, 0.001, err, result)
}

func TestMultilateration2DExact3(t *testing.T) {
	positions := [][]float64{
		{0.0, 0.0},
		{1000.0, 0.0},
		{0.0, 1000.0},
	}
	distances := []float64{math.Sqrt(2.0) * 1000.0, 1000.0, 1000.0}
	expected := []float64{1000.0, 1000.0}

	result, err := multilaterationSolver(positions, distances)
	testResults(t, expected, 0.001, err, result)
}

func multilaterationSolver(positions [][]float64, distances []float64) ([]float64, error) {
	solver := Multilateration{
		Positions: positions,
		Distances: distances,
	}

	p := optimize.Problem{
		Func: solver.Func,
		Grad: solver.Grad,
	}

	settings := optimize.DefaultSettings()
	if testing.Verbose() {
		settings.Recorder = recorder{}
	}
	settings.FunctionThreshold = 1e-6
	settings.GradientThreshold = 1e-6
	settings.FunctionConverge = nil

	result, err := optimize.Local(p, solver.InitialPoint(), settings, nil)

	return result.X, err
}

func testResults(t *testing.T, expected []float64, tolerance float64, err error, result []float64) {
	assert.NoError(t, err)
	assert.Equal(t, roundVector(expected, tolerance), roundVector(result, tolerance))
}

func roundVector(x []float64, unit float64) []float64 {
	rounded := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		rounded[i] = round(x[i], unit)
	}

	return rounded
}

func round(x, unit float64) float64 {
	return math.Round(x/unit) * unit
}
