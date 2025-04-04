import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;

public class
parametroEkorketaSVM_C_gamma {
    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            System.out.println("Uso: java SVM <train.arff> <dev.arff> <model.model> <parameters.txt>");
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
            System.out.println("\nDistribución de clases en Dev: ");
            printClassDistribution(dataDev);

            // Obtener el índice de la clase minoritaria
            int minorityClassIndex = getMinorityClassIndex(dataTrain);
            System.out.println("\nÍndice de la clase minoritaria: " + minorityClassIndex);

            // Parámetros a explorar
            double[] cValues = {0.1, 1, 10, 100, 1000};
            double[] gammaValues = {0.01, 0.1, 1, 10, 100};

            double bestFMeasure = 0;
            double bestC = 0;
            double bestGamma = 0;
            SMO bestModel = null;

            System.out.println("\nBúsqueda de hiperparámetros");
            for (double c : cValues) {
                for (double gamma : gammaValues) {
                    SMO svm = new SMO();

                    // Configurar parámetros
                    RBFKernel rbfKernel = new RBFKernel();
                    rbfKernel.setGamma(gamma);
                    svm.setKernel(rbfKernel);
                    svm.setC(c);

                    // Entrenar y evaluar (código existente...)
                    svm.buildClassifier(dataTrain);
                    Evaluation eval = new Evaluation(dataTrain);
                    eval.evaluateModel(svm, dataDev);

                    double f1Macro = 0.0;
                    for (int i = 0; i < dataTrain.numClasses(); i++) {
                        f1Macro += eval.fMeasure(i);
                    }
                    f1Macro /= dataTrain.numClasses();

                    System.out.println("C=" + c + ", Gamma=" + gamma + ", F1-macro=" + f1Macro);

                    if (!Double.isNaN(f1Macro) && f1Macro > bestFMeasure) {
                        bestFMeasure = f1Macro;
                        bestC = c;
                        bestGamma = gamma;
                        bestModel = svm;
                    }
                }
            }

            // Guardar el mejor modelo encontrado
            if (bestModel != null) {
                SerializationHelper.write(outModelPath, bestModel);
                System.out.println("Modelo guardado en: " + outModelPath);
            }

            System.out.println("Mejores parámetros: C=" + bestC +
                    ", Gamma=" + bestGamma);
            System.out.println("Mejor F-measure: " + bestFMeasure);

            // Guardar parámetros
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath));
            writer.write("Train=" + inTrainFSSPath + "\n");
            writer.write("Dev=" + inDevFSSPath + "\n");
            writer.write("Eredua=" + outModelPath + "\n");
            writer.write("\n");
            writer.write("C=" + bestC + "\n");
            writer.write("Gamma=" + bestGamma + "\n");
            writer.close();

            System.out.println("Parámetros guardados en: " + outputFilePath);

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
