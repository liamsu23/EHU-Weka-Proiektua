import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class parametroEkorketaSVM {
    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.out.println("Uso: java SVM <train_split_BOW_FSS.arff> <dev_split_BOW_FSS.arff> <SVM.model>");
            return;
        }
        try {
            String inTrainFSSPath = args[0];
            String inDevFSSPath = args[1];
            String outModelPath = args[2]; // Archivo donde se guardará el modelo

            // Cargar datasets
            DataSource source = new DataSource(inTrainFSSPath);
            Instances dataTrain = source.getDataSet();

            DataSource sourceDev = new DataSource(inDevFSSPath);
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
            int minorityClass = getMinorityClass(dataTrain);

            /*
            //-----PARAMETROS PARA OPTIMIZAR-----//
            -Estrategia recomendada para no demorar demasiado:
            1) Primero optimiza C y gamma (como ya haces).
            2) Prueba solo kernelType (RBF vs Lineal) → Si el lineal funciona bien, es más rápido.
            3) Si usas PolyKernel, prueba degree=2 o 3.
            4) Ajusta tol si el entrenamiento es muy lento (subir a 0.01 para mayor velocidad).

            -Tiempo vs Calidad del modelo:
            1) Si el tiempo es crítico: Quédate solo con C, gamma y prueba LinearKernel.
            2) Si puedes permitir más tiempo: Añade PolyKernel con degree=2 y ajusta tol.


            double[] cValues = {0.1, 1, 10, 100, 1000};
            double[] gammaValues = {0.0001, 0.001, 0.01, 0.1, 1, 10};
            String[] kernelTypes = {"RBF", "Linear"};  // RBF y Lineal (los más rápidos)

            double bestFMeasure = 0;
            double bestC = 0;
            double bestGamma = 0;
            String bestKernelType = "";
            SMO bestModel = null;

            for (double c : cValues) {
                for (double gamma : gammaValues) {
                    for (String kernelType : kernelTypes) {
                        // Solo aplicamos gamma si es RBF (el Lineal no lo usa)
                        if (kernelType.equals("Linear") && gamma != gammaValues[0]) {
                            continue;  // Evita repetir el LinearKernel para cada gamma
                        }

                        System.out.println("\nProbando: C=" + c + ", Gamma=" + gamma + ", Kernel=" + kernelType);

                        SMO svm = new SMO();

                        // Configurar kernel
                        if (kernelType.equals("RBF")) {
                            RBFKernel rbf = new RBFKernel();
                            rbf.setGamma(gamma);
                            svm.setKernel(rbf);
                        }
                        else if (kernelType.equals("Linear")) {
                            svm.setKernel(new weka.classifiers.functions.supportVector.LinearKernel());
                        }

                        svm.setC(c);
                        svm.setTolerance(0.001);  // Valor por defecto

                        // Entrenar y evaluar
                        svm.buildClassifier(dataTrain);
                        Evaluation eval = new Evaluation(dataTrain);
                        eval.evaluateModel(svm, dataDev);

                        // Obtener F-measure de la clase minoritaria
                        double fMeasure = eval.fMeasure(minorityClass);
                        System.out.println("F-measure: " + fMeasure);

                        // Actualizar mejor modelo
                        if (fMeasure > bestFMeasure) {
                            bestFMeasure = fMeasure;
                            bestC = c;
                            bestGamma = gamma;
                            bestKernelType = kernelType;
                            bestModel = svm;
                            System.out.println("¡Nuevo mejor modelo! (F-measure: " + bestFMeasure + ")");
                        }
                    }
                }
            }

            // Resultados finales
            System.out.println("\n● Mejores parámetros encontrados:");
            System.out.println("- Kernel: " + bestKernelType);
            System.out.println("- C: " + bestC);
            System.out.println("- Gamma: " + bestGamma);
            System.out.println("- F-measure: " + bestFMeasure);

            // Guardar el modelo final (opcional)
            SerializationHelper.write(outModelPath, bestModel);
            */

            // Parámetros a explorar
            double[] cValues = {0.1, 1, 10, 100, 1000};
            double[] gammaValues = {0.0001, 0.001, 0.01, 0.1, 1, 10};

            double bestFMeasure = 0;
            double bestC = 0;
            double bestGamma = 0;
            SMO bestModel = null;

            System.out.println("\nBúsqueda de hiperparámetros");
            // Búsqueda de hiperparámetros
            for (double c : cValues) {
                for (double gamma : gammaValues) {
                    System.out.println("Probando C = " + c + ", Gamma = " + gamma + "\n");

                    // Configurar SVM con kernel RBF y parámetros
                    SMO svm = new SMO();
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
                    double fMeasure = eval.fMeasure(minorityClass);
                    System.out.println("C = " + c + ", Gamma = " + gamma + ", F-measure = " + fMeasure);

                    // Guardar el mejor modelo encontrado
                    if (fMeasure > bestFMeasure) {
                        bestFMeasure = fMeasure;
                        bestC = c;
                        bestGamma = gamma;
                        bestModel = svm;
                    }
                }
            }

            // Resultados
            System.out.println("Mejores parámetros: C = " + bestC + ", Gamma = " + bestGamma);
            System.out.println("Mejor F-measure: " + bestFMeasure);

            // Guardar el mejor modelo encontrado
            SerializationHelper.write(outModelPath, bestModel);
            System.out.println("Modelo guardado en: " + outModelPath);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR: " + e.getMessage());
        }
    }

    // Método para obtener el índice de la clase minoritaria
    private static int getMinorityClass(Instances data) {
        int klaseMin = 0;
        int min = 0;
        for(int i = 0; i < data.numClasses(); i++){
            int maiztasuna = data.attributeStats(data.classIndex()).nominalCounts[i];
            if((maiztasuna < min) || i == 0){
                min = maiztasuna;
                klaseMin = i;
            }
        }
        System.out.println("Klase minoritario: " + klaseMin);
        return klaseMin;
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
        System.out.println("\n");
    }
}