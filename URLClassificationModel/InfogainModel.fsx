


open System
open Microsoft.ML
open Microsoft.ML.Data

printfn "\n\n\t\t\t-------------------- Infogain Data/Model --------------------\n"

//avgpathtokenlen,pathurlRatio,ArgUrlRatio,argDomanRatio,domainUrlRatio,pathDomainRatio,argPathRatio,CharacterContinuityRate,NumberRate_URL,NumberRate_FileName,NumberRate_AfterPath,Entropy_Domain,class
//105,0.876,0.008,0.087,0.095,9.174,0.01,0.435,0.199,0.219,-1,0.904,phishing
let dataPath = "./data/All_Infogain.csv"

[<CLIMutable>]
type URLData = {
  [<LoadColumn(0)>]
  avgpathtokenlen: float32

  [<LoadColumn(1)>]
  pathurlRatio: float32

  [<LoadColumn(2)>]
  ArgUrlRatio: float32

  [<LoadColumn(3)>]
  argDomanRatio: float32

  [<LoadColumn(4)>]
  domainUrlRatio: float32

  [<LoadColumn(5)>]
  pathDomainRatio: float32

  [<LoadColumn(6)>]
  argPathRatio: float32

  [<LoadColumn(7)>]
  CharacterContinuityRate: float32

  [<LoadColumn(8)>]
  NumberRate_URL: float32

  [<LoadColumn(9)>]
  NumberRate_FileName: float32
  
  [<LoadColumn(10)>]
  NumberRate_AfterPath: float32
  
  [<LoadColumn(11)>]
  Entropy_Domain: float32

  [<LoadColumn(12)>]
  URLclass: string
  }

[<CLIMutable>]
type URLPrediction = {
    PredictedLabel: string
}

let context = MLContext()

let data = 
  context
    .Data
    .LoadFromTextFile<URLData>(
      path = __SOURCE_DIRECTORY__ + dataPath,
      hasHeader = true,
      separatorChar = ',')

let datasets = context.Data.TrainTestSplit(data,testFraction=0.2)


let pipeline = 
    EstimatorChain()
        .Append(context.Transforms.NormalizeMinMax("avgpathtokenlen","avgpathtokenlen"))
        .Append(context.Transforms.NormalizeMinMax("pathurlRatio","pathurlRatio"))
        .Append(context.Transforms.NormalizeMinMax("ArgUrlRatio","ArgUrlRatio"))
        .Append(context.Transforms.NormalizeMinMax("argDomanRatio","argDomanRatio"))
        .Append(context.Transforms.NormalizeMinMax("domainUrlRatio","domainUrlRatio"))
        .Append(context.Transforms.NormalizeMinMax("pathDomainRatio","pathDomainRatio"))
        .Append(context.Transforms.NormalizeMinMax("argPathRatio","argPathRatio"))
        .Append(context.Transforms.NormalizeMinMax("CharacterContinuityRate","CharacterContinuityRate"))
        .Append(context.Transforms.NormalizeMinMax("NumberRate_URL","NumberRate_URL"))
        .Append(context.Transforms.NormalizeMinMax("NumberRate_FileName","NumberRate_FileName"))
        .Append(context.Transforms.NormalizeMinMax("NumberRate_AfterPath","NumberRate_AfterPath"))
        .Append(context.Transforms.NormalizeMinMax("Entropy_Domain","Entropy_Domain"))
        .Append(context.Transforms.Concatenate("Features",[|"avgpathtokenlen"; "pathurlRatio"; "ArgUrlRatio"; "argDomanRatio"; 
        "domainUrlRatio"; "pathDomainRatio"; "argPathRatio"; "CharacterContinuityRate";"CharacterContinuityRate";"NumberRate_URL";"NumberRate_FileName";"NumberRate_AfterPath";"Entropy_Domain";|]))
        .Append(context.Transforms.Conversion.MapValueToKey("Label","URLclass"))
        .AppendCacheCheckpoint(context) //cache data in memory

let postProcessingPipeline = 
    context.Transforms.Conversion.MapKeyToValue("PredictedLabel")

//ML .NET Multiclass Classification algorithms
let LbfgsMaximumEntropyAlgorithm = 
    context.MulticlassClassification.Trainers.LbfgsMaximumEntropy()

let SdcaMaximumEntropyAlgorithm = 
    context.MulticlassClassification.Trainers.SdcaMaximumEntropy()

let SdcaNonCalibratedAlgorithm = 
    context.MulticlassClassification.Trainers.SdcaNonCalibrated()

let NaiveBayesAlgorithm = 
    context.MulticlassClassification.Trainers.NaiveBayes()

let LightGbmAlgorithm = 
    context.MulticlassClassification.Trainers.LightGbm()


//set up pipeline for each
let LbfgsMaximumEntropyTrainingPipeline =
    pipeline
        .Append(LbfgsMaximumEntropyAlgorithm)
        .Append(postProcessingPipeline)

let SdcaMaximumEntropyPipeline =
    pipeline
        .Append(SdcaMaximumEntropyAlgorithm)
        .Append(postProcessingPipeline)

let SdcaNonCalibratedPipeline =
    pipeline
        .Append(SdcaNonCalibratedAlgorithm)
        .Append(postProcessingPipeline)

let NaiveBayesPipeline =
    pipeline
        .Append(NaiveBayesAlgorithm)
        .Append(postProcessingPipeline)

let LightGbmPipeline =
    pipeline
        .Append(LightGbmAlgorithm)
        .Append(postProcessingPipeline)

//create model for each
let LbfgsMaximumEntropyModel =
    datasets.TrainSet |> LbfgsMaximumEntropyTrainingPipeline.Fit

let SdcaMaximumEntropyModel =
    datasets.TrainSet |> SdcaMaximumEntropyPipeline.Fit

let SdcaNonCalibratedModel =
    datasets.TrainSet |> SdcaNonCalibratedPipeline.Fit

let NaiveBayesModel =
    datasets.TrainSet |> NaiveBayesPipeline.Fit

let LightGbmModel =
    datasets.TrainSet |> LightGbmPipeline.Fit


//evaulate and print each model
let LbfgsMaximumEntropyMetrics =
    (datasets.TestSet |> LbfgsMaximumEntropyModel.Transform)
    |> context.MulticlassClassification.Evaluate

let SdcaMaximumEntropyMetrics =
    (datasets.TestSet |> SdcaMaximumEntropyModel.Transform)
    |> context.MulticlassClassification.Evaluate

let SdcaNonCalibratedMetrics =
    (datasets.TestSet |> SdcaNonCalibratedModel.Transform)
    |> context.MulticlassClassification.Evaluate

let NaiveBayesMetrics =
    (datasets.TestSet |> NaiveBayesModel.Transform)
    |> context.MulticlassClassification.Evaluate

let LightGbmMetrics =
    (datasets.TestSet |> LightGbmModel.Transform)
    |> context.MulticlassClassification.Evaluate

printfn "\tAlgorithm \t\t\t\t Results\n"
printfn "LbfgsMaximumEntropy:\t Log Loss: %f \t Log Loss Reduction: %f \t\t MacroAccuracy: %f \t MicroAccuracy %f \n\n" LbfgsMaximumEntropyMetrics.LogLoss LbfgsMaximumEntropyMetrics.LogLossReduction LbfgsMaximumEntropyMetrics.MacroAccuracy LbfgsMaximumEntropyMetrics.MicroAccuracy 
printfn "SdcaMaximumEntropy:\t Log Loss: %f \t Log Loss Reduction: %f \t\t MacroAccuracy: %f \t MicroAccuracy %f \n\n" SdcaMaximumEntropyMetrics.LogLoss SdcaMaximumEntropyMetrics.LogLossReduction SdcaMaximumEntropyMetrics.MacroAccuracy SdcaMaximumEntropyMetrics.MicroAccuracy 
printfn "SdcaNonCalibrated:\t Log Loss: %f \t Log Loss Reduction: %f \t MacroAccuracy: %f \t MicroAccuracy %f \n\n" SdcaNonCalibratedMetrics.LogLoss SdcaNonCalibratedMetrics.LogLossReduction SdcaNonCalibratedMetrics.MacroAccuracy SdcaNonCalibratedMetrics.MicroAccuracy
printfn "NaiveBayes:\t\t Log Loss: %f \t Log Loss Reduction: %f \t MacroAccuracy: %f \t MicroAccuracy %f \n\n"  NaiveBayesMetrics.LogLoss NaiveBayesMetrics.LogLossReduction NaiveBayesMetrics.MacroAccuracy NaiveBayesMetrics.MicroAccuracy
printfn "LightGbm:\t\t Log Loss: %f \t Log Loss Reduction: %f \t\t MacroAccuracy: %f \t MicroAccuracy %f \n\n"  LightGbmMetrics.LogLoss LightGbmMetrics.LogLossReduction LightGbmMetrics.MacroAccuracy LightGbmMetrics.MicroAccuracy
printfn "\n\t\tConfusion Tables:\n\n"
printfn "Algorithm:\t LbfgsMaximumEntropy\t\n %A \n\n" <| LbfgsMaximumEntropyMetrics.ConfusionMatrix.GetFormattedConfusionTable()
printfn "Algorithm:\t SdcaMaximumEntropyMetrics\t\n %A \n\n" <| SdcaMaximumEntropyMetrics.ConfusionMatrix.GetFormattedConfusionTable()
printfn "Algorithm:\t SdcaNonCalibratedMetrics\t\n %A \n\n" <| SdcaNonCalibratedMetrics.ConfusionMatrix.GetFormattedConfusionTable()
printfn "Algorithm:\t NaiveBayesMetrics\t\n %A \n\n" <| NaiveBayesMetrics.ConfusionMatrix.GetFormattedConfusionTable()
printfn "Algorithm:\t LightGbmMetrics\t\n %A \n\n" <| LightGbmMetrics.ConfusionMatrix.GetFormattedConfusionTable()

printfn "\n\npress any key to exit..."
System.Console.ReadKey() |> ignore // keep console window open


// save model
//let savedModel = context.Model.Save(trainedModel, data.Schema, "model.zip");

//load model
//let (modelReloaded, schemaReloaded) = context.Model.Load("model.zip")

(* USE FOR MANUAL TESTING, IE HOSTING MODEL ON SERVER, change to correct features
let manualTest = {
     domain_token_count = 19
     executable = 0
     NumberofDotsinURL = 18
     Arguments_LongestWordLength= -1
     NumberRate_Domain = float32 0.041
     NumberRate_FileName = float32 0.656
     NumberRate_AfterPath = float32 -1
     Entropy_Domain = float32 0.612
     URLclass= ""
}

let predictionEngine = context.Model.CreatePredictionEngine<URLData, URLPrediction>(model)
let prediction = predictionEngine.Predict(manualTest)


printfn "Predicted Value: %s" prediction.PredictedLabel
*)