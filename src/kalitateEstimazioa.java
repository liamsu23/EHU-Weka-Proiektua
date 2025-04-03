import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;


/**
 * Klase honek SVM eredu baten kalitate-estimazio osoa egiten du, hainbat ebaluazio-metrika
 * aplikatuz ereduaren errendimendua neurtzeko. Prozesuak honako urratsak hartzen ditu:
 *
 * <p>Funtzionamendu nagusia:</p>
 * <ol>
 *   <li>Parametro optimoak kargatu aurrez definitutako fitxategitik (C, gamma, tolerance)</li>
 *   <li>Train eta dev datu-multzoak kargatu eta bateratu</li>
 *   <li>SVM eredua konfiguratu eta entrenatu datu guztiekin</li>
 *   <li>3 ebaluazio-metrika desberdin erabili:
 *     <ul>
 *       <li>Berrezarpen-akatsa (Resubstitution Error)</li>
 *       <li>10-fold ebaluazioa (Cross-Validation)</li>
 *       <li>Hold-out ebaluazio errepikatuak</li>
 *     </ul>
 *   </li>
 *   <li>Emaitzak fitxategian gorde</li>
 * </ol>
 *
 * <p>Erabilera:</p>
 * <pre>java kalitateEstimazioa train_split_BOW_FSS.arff dev_split_BOW_FSS.arff SVMopt.model parametroak.txt ebaluazioa.txt</pre>
 *
 * <p>Ebaluazio-metrikak:</p>
 * <ul>
 *   <li>Zehaztasuna (Accuracy)</li>
 *   <li>F1-Score (Macro)</li>
 *   <li>Errepikapen anitzetako emaitzen batezbestekoa</li>
 * </ul>
 *
 * <p>Ohar bereziak:</p>
 * <ul>
 *   <li>RBF kernela erabiltzen du SVMarentzat</li>
 *   <li>Dev multzoa erabiltzen du Hold-out ebaluaziorako</li>
 *   <li>5 errepikapen egiten ditu Hold-out ebaluaziorako</li>
 *   <li>Klase-atributua automatikoki detektatzen du (azken atributua)</li>
 * </ul>
 */

public class kalitateEstimazioa {
    public static void main(String[] args) throws Exception {

        if (args.length != 5) {
            System.out.println("Uso: java kalitateEstimazioa <train.arff> <dev.arff> <outputModel.model> <paramFile.txt> <ebaluazioa.txt>");
            return;
        }

        String inTrainFSSPath = args[0];   // Ruta del archivo de entrenamiento
        String inDevFSSPath = args[1];     // Ruta del archivo de validación (dev)
        String outModelPath = args[2];     // Ruta de salida para el modelo entrenado
        String paramFilePath = args[3];    // Ruta del archivo de parámetros óptimos
        String outEbaluazioakPath = args[4];   // Ruta de salida para las metricas obtenidas

        // Leer parámetros desde el archivo de texto
        double bestC = 0;
        double bestGamma = 0;
        double bestTolerance = 0;

        BufferedReader reader = new BufferedReader(new FileReader(paramFilePath));
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.startsWith("C=")) {
                bestC = Double.parseDouble(line.split("=")[1]);
            }
            else if (line.startsWith("Gamma=")) {
                bestGamma = Double.parseDouble(line.split("=")[1]);
            }
            else if (line.startsWith("Tolerance=")) {
                bestTolerance = Double.parseDouble(line.split("=")[1]);
            }
        }
        reader.close();

        System.out.println("Parámetros cargados: C=" + bestC + ", Gamma=" + bestGamma + ", Tolerance=" + bestTolerance);

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

        // Combinar los datasets de train y dev
        dataTrain.addAll(dataDev);  // Agrega todas las instancias de dataDev a dataTrain

        System.out.println("Número total de instancias (train + dev): " + dataTrain.numInstances());

        // Crear y configurar el modelo SVM
        SMO svm = new SMO();
        RBFKernel rbfKernel = new RBFKernel();
        rbfKernel.setGamma(bestGamma);
        svm.setKernel(rbfKernel);
        svm.setC(bestC);
        svm.setToleranceParameter(bestTolerance);

        // Entrenar el modelo con el conjunto de datos combinado (train + dev)
        svm.buildClassifier(dataTrain);

        // Guardar el modelo final
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
        writer.write("F1-Score (Macro): " + evalResubstitution.weightedFMeasure());

        // 2. 10-Fold Cross-Validation
        Evaluation evalCrossValidation = new Evaluation(dataTrain);
        evalCrossValidation.crossValidateModel(svm, dataTrain, 10, new java.util.Random(1)); // 10-fold
        System.out.println("\nEvaluación con 10-Fold Cross-Validation: ");
        System.out.println("Accuracy: " + evalCrossValidation.pctCorrect() + "%");
        System.out.println("F1-Score (Macro): " + evalCrossValidation.weightedFMeasure());
        writer.write("\nEbaluazioa 10-Fold Cross-Validation: ");
        writer.write("Accuracy: " + evalCrossValidation.pctCorrect() + "%");
        writer.write("F1-Score (Macro): " + evalCrossValidation.weightedFMeasure());

        // 3. Hold-out estratificado y repetido (opcional)
        // Esto es una evaluación Hold-out con división estratificada y repetida
        // Puede utilizarse como un conjunto de pruebas más robusto.
        Evaluation evalHoldOut = new Evaluation(dataTrain);
        evalHoldOut.evaluateModel(svm, dataDev);
        System.out.println("\nEvaluación Hold-Out (con dev): ");
        System.out.println("Accuracy: " + evalHoldOut.pctCorrect() + "%");
        System.out.println("F1-Score (Macro): " + evalHoldOut.weightedFMeasure());
        writer.write("\nEbaluazio Hold-Out (dev multzoarekin): ");
        writer.write("Accuracy: " + evalHoldOut.pctCorrect() + "%");
        writer.write("F1-Score (Macro): " + evalHoldOut.weightedFMeasure());

        // Si deseas repetir el Hold-out y calcular un promedio de las métricas
        // puedes realizar varias repeticiones (opcional):
        // Repeticiones del Hold-Out para obtener un intervalo de confianza
        int numRepetitions = 5;
        double totalAccuracy = 0;
        double totalFMeasure = 0;
        for (int i = 0; i < numRepetitions; i++) {
            evalHoldOut.evaluateModel(svm, dataDev);
            totalAccuracy += evalHoldOut.pctCorrect();
            totalFMeasure += evalHoldOut.weightedFMeasure();
        }
        double avgAccuracy = totalAccuracy / numRepetitions;
        double avgFMeasure = totalFMeasure / numRepetitions;
        System.out.println("\nEvaluación Hold-Out (promedio de " + numRepetitions + " repeticiones):");
        System.out.println("Accuracy promedio: " + avgAccuracy + "%");
        System.out.println("F1-Score (Macro) promedio: " + avgFMeasure);
        writer.write("\nEbaluazio Hold-Out (bb errepikapenak: " + numRepetitions + "):");
        writer.write("Accuracy bb: " + avgAccuracy + "%");
        writer.write("F1-Score (Macro) bb: " + avgFMeasure);

        writer.close();
    }
}
