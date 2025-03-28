import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;

public class parametroEkorketaSVM_2 {
    public static void main(String[] args) throws Exception {

        if (args.length <4) {
            System.out.println("Uso: java SVM <train.arff> <dev.arff> <model.model>");
            return;
        }
        try {
            String inTrainFSSPath = args[0];
            String inDevFSSPath = args[1];
            String outModelPath = args[2]; // Archivo donde se guardará el modelo
            String outputFilePath = args[3]; // Archivo donde se guardarán los mejores parámetros

            // Cargar datasets
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(inTrainFSSPath);
            Instances dataTrain = source.getDataSet();

            ConverterUtils.DataSource sourceDev = new ConverterUtils.DataSource(inDevFSSPath);
            Instances dataDev = sourceDev.getDataSet();

            // Establecer la clase objetivo (última columna)
            if (dataTrain.classIndex() == -1) {
                dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
            }
            if (dataDev.classIndex() == -1) {
                dataDev.setClassIndex(dataDev.numAttributes() - 1);
            }

            // Imprimir distribución de clases
            System.out.println("Distribución de clases en Train: ");
            printClassDistribution(dataTrain);
            System.out.println("Distribución de clases en Dev: ");
            printClassDistribution(dataDev);

            // Obtener el índice de la clase minoritaria
            int minorityClassIndex = getMinorityClassIndex(dataTrain);
            System.out.println("Índice de la clase minoritaria: " + minorityClassIndex);

            // Parámetros a explorar
            double[] cValues = {1, 10, 100, 1000};
            double[] gammaValues = {0.01, 0.1, 1};
            // String[] kernelValues = {"rbf", "poly"};
            double[] epsilonValues = {0.0001, 0.001, 0.01, 0.1};

            double bestFMeasure = 0;
            double bestC = 0;
            double bestGamma = 0;
            double bestEpsilon = 0;
            SMO bestModel = null;

            System.out.println("🔍 Búsqueda de hiperparámetros");
            // Búsqueda de hiperparámetros
            for (double c : cValues) {
                System.out.println("Probando C=" + c);
                for (double gamma : gammaValues) {
                    System.out.println("Probando Gamma=" + gamma);
                    for (double epsilon : epsilonValues){
                        System.out.println("Probando Kernel= " + epsilon);

                        SMO svm = new SMO();
                        RBFKernel rbfKernel = new RBFKernel();
                        rbfKernel.setGamma(gamma);
                        svm.setKernel(rbfKernel);
                        svm.setC(c);
                        svm.setEpsilon(epsilon);


                        // Entrenar con train
                        svm.buildClassifier(dataTrain);

                        // Evaluar en dev
                        Evaluation eval = new Evaluation(dataTrain);
                        eval.evaluateModel(svm, dataDev);

                        // Obtener accuracy
                        double accuracy = eval.pctCorrect() / 100.0;

                        // Obtener F1-score macro (promedio de F1-score por clase)
                        double f1Macro = 0.0;
                        for (int i = 0; i < dataTrain.numClasses(); i++) {
                            f1Macro += eval.fMeasure(i);
                        }
                        f1Macro /= dataTrain.numClasses();

                        System.out.println("C=" + c + ", Gamma=" + gamma + ", Accuracy=" + accuracy + ", F1-macro=" + f1Macro);

                        // Guardar el mejor modelo basado en F1-macro
                        if (!Double.isNaN(f1Macro) && f1Macro > bestFMeasure) {
                            bestFMeasure = f1Macro; // Ahora el criterio es F1-macro
                            bestC = c;
                            bestGamma = gamma;
                            bestEpsilon = epsilon;
                            bestModel = svm;
                        }
                    }
                }
            }

            // Guardar el mejor modelo encontrado
            if (bestModel != null) {
                SerializationHelper.write(outModelPath, bestModel);
                System.out.println("✅ Modelo guardado en: " + outModelPath);
            }

            System.out.println("Mejores parámetros: C=" + bestC + ", Gamma=" + bestGamma);
            System.out.println("Mejor F-measure: " + bestFMeasure);

            // Guardar los parámetros en un archivo de texto
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath));
            writer.write("C=" + bestC + "\n");
            writer.write("Gamma=" + bestGamma + "\n");
            writer.write("Epsilon=" + bestEpsilon + "\n");
            writer.close();

            System.out.println("Parámetros guardados en el archivo: " + outputFilePath);

        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR: " + e.getMessage());
        }
    }

    // Método para obtener el índice de la clase minoritaria
    private static int getMinorityClassIndex(Instances data) {
        int numClasses = data.numClasses();
        int[] classCounts = new int[numClasses];

        for (int i = 0; i < data.numInstances(); i++) {
            int classIndex = (int) data.instance(i).classValue();
            classCounts[classIndex]++;
        }

        int minIndex = 0;
        for (int i = 1; i < numClasses; i++) {
            if (classCounts[i] < classCounts[minIndex]) {
                minIndex = i;
            }
        }
        return minIndex;
    }

    // Método para imprimir la distribución de clases en el dataset
    private static void printClassDistribution(Instances data) {
        int numClasses = data.numClasses();
        int[] classCounts = new int[numClasses];

        for (int i = 0; i < data.numInstances(); i++) {
            int classIndex = (int) data.instance(i).classValue();
            classCounts[classIndex]++;
        }

        for (int i = 0; i < numClasses; i++) {
            String className = data.classAttribute().value(i);
            System.out.println("Clase " + i + " -> " + className + ": " + classCounts[i] + " instancias");
        }
    }
}