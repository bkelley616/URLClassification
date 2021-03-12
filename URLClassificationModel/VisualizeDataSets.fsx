open FSharp.Data
open XPlot.Plotly


[<Literal>]
let BestFirstDataPath = "./data/All_BestFirst.csv"

[<Literal>]
let InfogainDataPath = "./data/All_Infogain.csv"

type BestFirstDataSet = CsvProvider<BestFirstDataPath>
type InfogainDataSet = CsvProvider<InfogainDataPath>

let BestFirstData = BestFirstDataSet.Load( __SOURCE_DIRECTORY__ + BestFirstDataPath)
let InfogainData = InfogainDataSet.Load( __SOURCE_DIRECTORY__ + InfogainDataPath)

printfn "BestFirstData Features:\n\n%A\n\n" BestFirstData.Headers.Value
printfn "InfogainData Features:\n\n%A\n" InfogainData.Headers.Value
printfn "\n\n\n"

let BestFirstCountByLabel = 
    BestFirstData.Rows 
    |> Seq.groupBy( fun row -> row.Class)
    |> Seq.map( fun (label, rows) -> (label, rows |> Seq.length))

let InfogainCountByLabel = 
    InfogainData.Rows 
    |> Seq.groupBy( fun row -> row.Class)
    |> Seq.map( fun (label, rows) -> (label, rows |> Seq.length))

let BestFirstDataPieGraph =
    BestFirstCountByLabel
    |> Chart.Pie
    |> Chart.WithTitle "All_BestFirst.csv"
    |> Chart.WithLegend true

let InfogainDataPieGraph =
    InfogainCountByLabel
    |> Chart.Pie
    |> Chart.WithTitle "All_Infogain.csv"
    |> Chart.WithLegend true

let bestFirstX =
    BestFirstCountByLabel |> Seq.map(fun (label, _) -> label )

let bestFirstY = 
    BestFirstCountByLabel |> Seq.map(fun (_, count) -> count )

let infogainX =
    InfogainCountByLabel |> Seq.map(fun (label, _) -> label )

let infogainY = 
    InfogainCountByLabel |> Seq.map(fun (_, count) -> count )

let groupedTrace1 =
    Bar(
        x = bestFirstX,
        y = bestFirstY,
        name= "All_BestFirst.csv"
    )

let groupedTrace2 =
    Bar(
        x = infogainX,
        y = infogainY,
        name = "All_Infogain.csv"
    )

let groupedLayout = Layout(barmode = "group")

let ComparisonBarChart =
    [groupedTrace1; groupedTrace2]
    |> Chart.Plot
    |> Chart.WithLayout groupedLayout
    |> Chart.WithTitle "BestFirst.csv vs Infogain.csv - Count by URL Type"
    |> Chart.WithYTitle "Count"
    |> Chart.WithXTitle "URL Type"


seq { BestFirstDataPieGraph; InfogainDataPieGraph; ComparisonBarChart} |> Chart.ShowAll
