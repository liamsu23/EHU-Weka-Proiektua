import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.*;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

public class kalitateEstimazioa_3 {
    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            System.out.println("Uso: java kalitateEstimazioa <train.arff> <dev.arff> <outputModel.model> <paramFile.txt> <ebaluazioa.txt>");
            return;
        }

        String inTrainFSSPath = args[0];   // Ruta del archivo de entrenamiento
        String inDevFSSPath = args[1];     // Ruta del archivo de validación (dev)
        String outModelPath = args[2];     // Ruta de salida para el modelo entrenado
        String paramFilePath = args[3];    // Ruta del archivo de parámetros óptimos
        String outEbaluazioakPath = args[4];   // Ruta de salida para las métricas obtenidas

        // Leer parámetros desde el archivo de texto
        double bestC = 0;
        double bestGamma = 0;
        double bestDegree = 0;  // Para PolyKernel
        double bestOmega = 0;   // Para PukKernel
        double bestSigma = 0;   // Para PukKernel
        String bestKernel = ""; // El tipo de Kernel que se ha seleccionado

        BufferedReader reader = new BufferedReader(new FileReader(paramFilePath));
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.startsWith("C=")) {
                bestC = Double.parseDouble(line.split("=")[1]);
            }
            else if (line.startsWith("Gamma=")) {
                bestGamma = Double.parseDouble(line.split("=")[1]);
            }
            else if (line.startsWith("Degree=")) {
                bestDegree = Double.parseDouble(line.split("=")[1]);
            }
            else if (line.startsWith("Omega=")) {
                bestOmega = Double.parseDouble(line.split("=")[1]);
            }
            else if (line.startsWith("Sigma=")) {
                bestSigma = Double.parseDouble(line.split("=")[1]);
            }
            else if (line.startsWith("Kernel=")) {
                bestKernel = line.split("=")[1];
            }
        }
        reader.close();

        System.out.println("Parámetros cargados: C=" + bestC + ", Gamma=" + bestGamma + ", Kernel=" + bestKernel);

        // Cargar dataset de entrenamiento
        ConverterUtils.DataSource sourceTrain = new ConverterUtils.DataSource(inTrainFSSPath);
        Instances dataTrain = sourceTrain.getDataSet();

        // Cargar dataset de validación (dev)
        ConverterUtils.DataSource sourceDev = new ConverterUtils.DataSource(inDevFSSPath);
        Instances dataDev = sourceDev.getDataSet();

        // Establecer la clase objetivo (última columna) en ambos datasets
        if (dataTrain.classIndex() == -1) {
            dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
        }
        if (dataDev.classIndex() == -1) {
            dataDev.setClassIndex(dataDev.numAttributes() - 1);
        }

        // Crear y configurar el modelo SVM
        SMO svm = new SMO();
        Kernel kernel = null;

        // Configurar el kernel según lo especificado en los parámetros
        switch (bestKernel) {
            case "RBFKernel":
                kernel = new RBFKernel();
                ((RBFKernel) kernel).setGamma(bestGamma);
                break;
            case "PolyKernel":
                kernel = new PolyKernel();
                ((PolyKernel) kernel).setExponent(bestDegree);
                break;
            case "Puk":
                kernel = new Puk();
                ((Puk) kernel).setOmega(bestOmega);
                ((Puk) kernel).setSigma(bestSigma);
                break;
            default:
                System.out.println("Kernel no reconocido. Usando RBFKernel por defecto.");
                kernel = new RBFKernel();
                ((RBFKernel) kernel).setGamma(bestGamma);
        }

        svm.setKernel(kernel);
        svm.setC(bestC);

        // Entrenar el modelo con el conjunto de datos combinado (train + dev)
        System.out.println("Entrenando el modelo SVM...");
        svm.buildClassifier(dataTrain);

        // Guardar el modelo final
        System.out.println("Guardando el modelo entrenado...");
        SerializationHelper.write(outModelPath, svm);
        System.out.println("Modelo final guardado en: " + outModelPath);

        // Evaluaciones
        BufferedWriter writer = new BufferedWriter(new FileWriter(outEbaluazioakPath));

        // 1. Resubstitution Error (Evaluación en el mismo conjunto de entrenamiento + validación)
        Evaluation evalResubstitution = new Evaluation(dataTrain);
        evalResubstitution.evaluateModel(svm, dataTrain);
        System.out.println("\nEvaluación de Resustitución: ");
        System.out.println("Accuracy: " + evalResubstitution.pctCorrect() + "%");
        System.out.println("F1-Score (Macro): " + evalResubstitution.weightedFMeasure());
        writer.write("Ebaluazioa ez-zintzoa (Resubstitution Error): ");
        writer.write("Accuracy: " + evalResubstitution.pctCorrect() + "%");
        writer.write(" F1-Score (Macro): " + evalResubstitution.weightedFMeasure() + "\n");

        // 2. 10-Fold Cross-Validation
        Evaluation evalCrossValidation = new Evaluation(dataTrain);
        evalCrossValidation.crossValidateModel(svm, dataTrain, 10, new java.util.Random(1)); // 10-fold
        System.out.println("\nEvaluación con 10-Fold Cross-Validation: ");
        System.out.println("Accuracy: " + evalCrossValidation.pctCorrect() + "%");
        System.out.println("F1-Score (Macro): " + evalCrossValidation.weightedFMeasure());
        writer.write("\nEbaluazioa 10-Fold Cross-Validation: ");
        writer.write("Accuracy: " + evalCrossValidation.pctCorrect() + "%");
        writer.write(" F1-Score (Macro): " + evalCrossValidation.weightedFMeasure() + "\n");

        // 3. Hold-out estratificado
        Evaluation evalHoldOut = new Evaluation(dataTrain);
        evalHoldOut.evaluateModel(svm, dataDev);
        System.out.println("\nEvaluación Hold-Out (con dev): ");
        System.out.println("Accuracy: " + evalHoldOut.pctCorrect() + "%");
        System.out.println("F1-Score (Macro): " + evalHoldOut.weightedFMeasure());
        writer.write("\nEbaluazio Hold-Out (dev multzoarekin): ");
        writer.write("Accuracy: " + evalHoldOut.pctCorrect() + "%");
        writer.write(" F1-Score (Macro): " + evalHoldOut.weightedFMeasure() + "\n");

        writer.close();
    }
}
