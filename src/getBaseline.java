import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SimpleLogistic;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;


/**
 * Klase honek eredu basiko bat (baseline) eraikitzen du SimpleLogistic erabiliz,
 * machine learning proiektu baten oinarri gisa balio duena.
 *
 * <p>Funtzionamendu nagusia:</p>
 * <ol>
 *   <li>Datu-multzoak kargatu (train eta dev FSS formatuan)</li>
 *   <li>SimpleLogistic eredua entrenatu</li>
 *   <li>Eredua ebaluatu dev multzoan</li>
 *   <li>Emaitzak gorde (terminalean eta fitxategian)</li>
 *   <li>Predikzioak gorde fitxategian</li>
 * </ol>
 *
 * <p>Erabilera:</p>
 * <pre>java getBaseline train_BOW_FSS.arff dev_BOW_FSS.arff ebaluazioaLR.txt</pre>
 *
 * <p>Ebaluazio-metrikak:</p>
 * <ul>
 *   <li>Klasearen araberako zehaztasuna (precision, recall, F1)</li>
 *   <li>Nahasmen-matrizia</li>
 *   <li>Errore metrikoak (MAE, RMSE)</li>
 * </ul>
 *
 * <p>Oharrak:</p>
 * <ul>
 *   <li>SimpleLogistic: Logistika-erregresio sinple eta azkarra</li>
 *   <li>Klase-atributua automatikoki detektatzen du (azken atributua)</li>
 *   <li>Emaitzak bai terminalean bai fitxategian gordetzen dira</li>
 * </ul>
 */

public class getBaseline {

    public static void main(String[] args) throws Exception {

        if (args.length != 5) {
            System.out.println("Uso: java LogisticRegression <train_split_BOW_FSS.arff> <dev_split_BOW_FSS.arff> <test_BOW_FSS.arff> <ebaluazioLR.txt> <predikzioLR.txt");
            return;
        }

        try {
            String trainPath = args[0];
            String devPath= args[1];
            String testPath = args[2];
            String outEbaluazioakPath = args[3];
            String outPredictionsPath = args[4];

            // Cargar datasets
            ConverterUtils.DataSource sourceTrain = new ConverterUtils.DataSource(trainPath);
            Instances dataTrain = sourceTrain.getDataSet();

            ConverterUtils.DataSource sourceDev = new ConverterUtils.DataSource(devPath);
            Instances dataDev = sourceDev.getDataSet();

            ConverterUtils.DataSource sourceTest = new ConverterUtils.DataSource(testPath);
            Instances dataTest = sourceDev.getDataSet();

            // Establecer la variable objetivo (última columna)
            if (dataTrain.classIndex() == -1) {
                dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
            }
            if (dataDev.classIndex() == -1) {
                dataDev.setClassIndex(dataDev.numAttributes() - 1);
            }
            if (dataTest.classIndex() == -1) {
                dataTest.setClassIndex(dataDev.numAttributes() - 1);
            }

            // 3. Crear el modelo de Regresión Logística
            Classifier model = new SimpleLogistic();
            //Classifier model = new Logistic();
            System.out.println("Entrenando modelo...");
            model.buildClassifier(dataTrain);

            // 4. Evaluar el modelo con el conjunto de desarrollo (dev)
            Evaluation eval = new Evaluation(dataTrain);
            System.out.println("Evaluando modelo...");
            eval.evaluateModel(model, dataDev);

            BufferedWriter writer = new BufferedWriter(new FileWriter(outEbaluazioakPath));

            // 5. Imprimir resultados de la evaluación
            System.out.println("=== Evaluación del Modelo con Dev ===");
            System.out.println(eval.toSummaryString());
            System.out.println("=== Matriz de Confusión ===");
            System.out.println(eval.toMatrixString());
            System.out.println("=== Medidas de Precisión ===");
            System.out.println(eval.toClassDetailsString());

            writer.write("=== Dev modeloaren ebaluazioa ===" + "\n");
            writer.write(eval.toSummaryString());
            writer.write("=== Nahasmen matrizea ===" + "\n");
            writer.write(eval.toMatrixString());
            writer.write("=== Precision ===" + "\n");
            writer.write(eval.toClassDetailsString());

            // Imprimir métricas de regresión
            System.out.println("\nError absoluto medio (MAE): " + eval.meanAbsoluteError());
            System.out.println("\nRaíz del error cuadrático medio (RMSE): " + eval.rootMeanSquaredError());

            writer.write("\nMean Absolute Error (MAE): " + eval.meanAbsoluteError());
            writer.write("\nRoot Mean Squared Error (RMSE): " + eval.rootMeanSquaredError());
            writer.close();


            BufferedWriter predWriter = new BufferedWriter(new FileWriter(outPredictionsPath));
            predWriter.write("Instancia,Predicción\n");

            // Realizar predicciones para cada instancia en el conjunto test
            for (int i = 0; i < dataTest.numInstances(); i++) {
                // Obtener la instancia a predecir
                double classLabel = model.classifyInstance(dataTest.instance(i));

                // Convertir el número de clase a su nombre
                String className = dataTest.classAttribute().value((int) classLabel);

                // Escribir el ID de la instancia y la clase predicha en el archivo de salida
                predWriter.write(i + "," + className + "\n");
            }
            predWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR: " + e.getMessage());
        }
    }
}
