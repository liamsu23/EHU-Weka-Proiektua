import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.*;

public class getModel {

    public static void main(String[] args) throws Exception {

        if (args.length != 9) {
            System.out.println("Uso: java getModel <train_split_BOW_FSS.arff> <dev_split_BOW_FSS.arff> <test_BOW_FSS.arff> <ebaluazioLR.txt> <predikzioLR.txt> <SVM.model> <parameters.txt> <SVMopt.model> <ebaluazioSVM.txt>");
            return;
        }

        try {
            // Argumentos para getBaseline
            String trainPath = args[0];
            String devPath = args[1];
            String testPath = args[2];
            String outEvaluationLRPath = args[3];
            String outPredictionsPath = args[4];

            // Argumentos para parametroEkorketa
            //String inTrainFSSPath = args[0]; el mismo que args[0]
            //String inDevFSSPath = args[1]; el mismo que args[1]
            String outModelPath = args[5];
            String outParametersPath = args[6];

            // Argumentos para kalitateEstimazioa
            //String inTrainFSSPath = args[0]; el mismo que args[0]
            //String inDevFSSPath = args[1]; el mismo que args[1]
            String outModelOptPath = args[7];
            //String paramFilePath = args[3]; el mismo que args[6]
            String outEvaluationSVMPath = args[8];


            Instances dataTrain = loadDataset(trainPath);
            Instances dataDev = loadDataset(devPath);
            Instances dataTest = loadDataset(testPath);

            // === PASO 1: Entrenar y evaluar modelo de línea base (Regresión Logística) ===
            getBaseline(dataTrain, dataDev, dataTest, outEvaluationLRPath, outPredictionsPath);

            // === PASO 2: Búsqueda de parámetros para SVM ===
            parametroEkorketa(dataTrain, dataDev, outModelPath, outParametersPath);

            // === PASO 3: Entrenamiento final y evaluación del SVM ===
            trainFinalSVM(dataTrain, dataDev, outModelOptPath, outParametersPath, outEvaluationSVMPath);

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR: " + e.getMessage());
        }
    }

    // ============== Métodos para Regresión Logística (getModel) ==============
    private static void getBaseline(Instances train, Instances dev, Instances test, String outEvaluationPath, String outPredictionsPath) throws Exception {
        System.out.println("\n=== Entrenando modelo de Regresión Logística ===");

        // Crear y entrenar el modelo
        Classifier model = new SimpleLogistic();
        System.out.println("Entrenando modelo...");
        model.buildClassifier(train);

        // Evaluar el modelo
        Evaluation eval = evaluateModel(model, train, dev);
        saveEvaluationResults(eval, outEvaluationPath);
        savePredictions(model, test, outPredictionsPath);
    }

    // ============== Métodos para SVM (parametroEkorketaSVM) ==============
    private static void parametroEkorketa(Instances train, Instances dev, String outModelPath, String outParametersPath) throws Exception {
        System.out.println("\n=== Buscando mejores parámetros para SVM ===");

        // Parámetros a explorar
        double[] cValues = {0.1, 1, 10, 100, 1000};
        double[] gammaValues = {0.01, 0.1, 1, 10, 100};
        double[] toleranceValues = {1.0e-3, 1.0e-4, 1.0e-5};

        double bestFMeasure = 0;
        double bestC = 0;
        double bestGamma = 0;
        double bestTolerance = 1.0e-3;
        SMO bestModel = null;

        for (double c : cValues) {
            for (double gamma : gammaValues) {
                for (double tolerance : toleranceValues) {
                    SMO svm = new SMO();
                    RBFKernel rbfKernel = new RBFKernel();
                    rbfKernel.setGamma(gamma);
                    svm.setKernel(rbfKernel);
                    svm.setC(c);
                    svm.setToleranceParameter(tolerance);

                    svm.buildClassifier(train);
                    Evaluation eval = new Evaluation(train);
                    eval.evaluateModel(svm, dev);

                    double f1Macro = calculateMacroF1(eval, train.numClasses());
                    System.out.printf("C=%.1f, Gamma=%.2f, Tol=%.1e, F1-macro=%.4f%n",
                            c, gamma, tolerance, f1Macro);

                    if (!Double.isNaN(f1Macro) && f1Macro > bestFMeasure) {
                        bestFMeasure = f1Macro;
                        bestC = c;
                        bestGamma = gamma;
                        bestTolerance = tolerance;
                        bestModel = svm;
                    }
                }
            }
        }

        if (bestModel != null) {
            SerializationHelper.write(outModelPath, bestModel);
            saveSVMParameters(outParametersPath, outModelPath, bestC, bestGamma, bestTolerance);
        }
    }

    // ============== Métodos para evaluación final (kalitateEstimazioa) ==============
    private static void trainFinalSVM(Instances train, Instances dev, String outModelPath, String paramsPath, String outEvaluationPath) throws Exception {
        System.out.println("\n=== Entrenando modelo SVM final ===");

        // Leer parámetros óptimos
        SVMParameters params = readSVMParameters(paramsPath);

        // Cargar y combinar datasets
        train.addAll(dev);

        // Crear y entrenar modelo final
        SMO svm = new SMO();
        RBFKernel rbfKernel = new RBFKernel();
        rbfKernel.setGamma(params.gamma);
        svm.setKernel(rbfKernel);
        svm.setC(params.c);
        svm.setToleranceParameter(params.tolerance);
        svm.buildClassifier(train);

        // Guardar modelo final
        SerializationHelper.write(outModelPath, svm);

        // Evaluar modelo
        evaluateFinalModel(svm, train, dev, outEvaluationPath);
    }

    // ============== Métodos auxiliares ==============
    private static Instances loadDataset(String path) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    private static Evaluation evaluateModel(Classifier model, Instances trainData, Instances testData) throws Exception {
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(model, testData);
        return eval;
    }

    private static void saveEvaluationResults(Evaluation eval, String path) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path))) {
            writer.write("=== Evaluación del modelo ===\n");
            writer.write(eval.toSummaryString());
            writer.write("\n=== Matriz de confusión ===\n");
            writer.write(eval.toMatrixString());
            writer.write("\n=== Métricas por clase ===\n");
            writer.write(eval.toClassDetailsString());
            writer.write("\nMAE: " + eval.meanAbsoluteError());
            writer.write("\nRMSE: " + eval.rootMeanSquaredError());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void savePredictions(Classifier model, Instances data, String path) throws Exception {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path))) {
            writer.write("Instancia,Predicción\n");
            for (int i = 0; i < data.numInstances(); i++) {
                double prediction = model.classifyInstance(data.instance(i));
                String className = data.classAttribute().value((int) prediction);
                writer.write(i + "," + className + "\n");
            }
        }
    }

    private static double calculateMacroF1(Evaluation eval, int numClasses) {
        double f1Sum = 0;
        for (int i = 0; i < numClasses; i++) {
            f1Sum += eval.fMeasure(i);
        }
        return f1Sum / numClasses;
    }

    private static void saveSVMParameters(String path, String modelPath, double c, double gamma, double tolerance) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path))) {
            writer.write("Model=" + modelPath + "\n\n");
            writer.write("C=" + c + "\n");
            writer.write("Gamma=" + gamma + "\n");
            writer.write("Tolerance=" + tolerance + "\n");
        }
    }

    private static SVMParameters readSVMParameters(String path) throws IOException {
        SVMParameters params = new SVMParameters();
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("C=")) {
                    params.c = Double.parseDouble(line.substring(2));
                } else if (line.startsWith("Gamma=")) {
                    params.gamma = Double.parseDouble(line.substring(6));
                } else if (line.startsWith("Tolerance=")) {
                    params.tolerance = Double.parseDouble(line.substring(10));
                }
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return params;
    }

    private static void evaluateFinalModel(SMO model, Instances trainData, Instances devData, String outputPath) throws Exception {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            // 1. Resubstitution Error
            Evaluation evalTrain = new Evaluation(trainData);
            evalTrain.evaluateModel(model, trainData);
            writer.write("=== Evaluación en Train+Dev ===\n");
            writer.write(evalTrain.toSummaryString());

            // 2. Cross-validation
            Evaluation evalCV = new Evaluation(trainData);
            evalCV.crossValidateModel(model, trainData, 10, new java.util.Random(1));
            writer.write("\n=== 10-Fold Cross-Validation ===\n");
            writer.write(evalCV.toSummaryString());

            // 3. Hold-out (si hay datos de dev separados)
            if (devData.numInstances() > 0) {
                Evaluation evalDev = new Evaluation(trainData);
                evalDev.evaluateModel(model, devData);
                writer.write("\n=== Evaluación en Dev ===\n");
                writer.write(evalDev.toSummaryString());
            }
        }
    }

    // Clase auxiliar para almacenar parámetros SVM
    private static class SVMParameters {
        double c;
        double gamma;
        double tolerance;
    }
}
