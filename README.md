# Multilateration
Multilateration for use with gonum

## Usage
```go
import(
	"github.com/dordille/mulilateration"
	"gonum.org/v1/gonum/optimize"
)

func main() {
	positions := [][]float64{
		{1.0, 1.0},
		{3.0, 1.0},
		{2.0, 2.0},
	}
	distances := []float64{1.0, 1.0, 1.0}
	solver := Multilateration{
		Positions: positions,
		Distances: distances,
	}

	p := optimize.Problem{
		Func: solver.Func,
		Grad: solver.Grad,
	}

	settings := optimize.DefaultSettings()
	result, err := optimize.Local(p, solver.InitialPoint(), settings, nil)
}

```
