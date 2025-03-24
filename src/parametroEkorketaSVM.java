import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

public class parametroEkorketaSVM {
    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.out.println("Uso: java SVM <train.arff> <dev.arff> <model.model>");
            return;
        }
        try {
            String inTrainFSSPath = args[0];
            String inDevFSSPath = args[1];
            String outModelPath = args[2]; // Archivo donde se guardará el modelo

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
            double[] gammaValues = {0.001, 0.01, 0.1, 1};

            double bestFMeasure = 0;
            double bestC = 0;
            double bestGamma = 0;
            SMO bestModel = null;

            System.out.println("🔍 Búsqueda de hiperparámetros");
            // Búsqueda de hiperparámetros
            for (double c : cValues) {
                System.out.println("Probando C=" + c);
                for (double gamma : gammaValues) {
                    System.out.println("Probando Gamma=" + gamma);
                    SMO svm = new SMO();

                    // Configurar SVM con kernel RBF y parámetros
                    RBFKernel rbfKernel = new RBFKernel();
                    rbfKernel.setGamma(gamma);
                    svm.setKernel(rbfKernel);
                    svm.setC(c);

                    // Entrenar con train
                    svm.buildClassifier(dataTrain);

                    // Evaluar en dev
                    Evaluation eval = new Evaluation(dataTrain);
                    eval.evaluateModel(svm, dataDev);

                    // Obtener F-measure de la clase minoritaria
                    double fMeasure = eval.fMeasure(minorityClassIndex);
                    System.out.println("C=" + c + ", Gamma=" + gamma + ", F-measure=" + fMeasure);

                    // Guardar el mejor modelo encontrado
                    if (!Double.isNaN(fMeasure) && fMeasure > bestFMeasure) {
                        bestFMeasure = fMeasure;
                        bestC = c;
                        bestGamma = gamma;
                        bestModel = svm;
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
