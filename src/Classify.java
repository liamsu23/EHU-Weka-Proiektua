import java.io.BufferedWriter;
import java.io.FileWriter;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;

public class Classify {
    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.out.println("Erabilera : java -jar Classify.jar <SMVopt.model> <test_BOW_FSS.arff> <predictions.txt>");
            return;
        }

        String inputModelFilePath = args[0];            // Ruta del modelo entrenado
        String inputTestFSSFilePath = args[1];          // Ruta del archivo de datos de prueba
        String outputFilePath = args[2];                // Ruta del archivo de salida para las predicciones

        // Cargar el modelo previamente entrenado
        SMO svm = (SMO) SerializationHelper.read(inputModelFilePath);
        System.out.println("Modeloa kargatu da hurrengo helbidetik: " + inputModelFilePath);

        // Cargar el conjunto de datos de prueba
        ConverterUtils.DataSource sourceTest = new ConverterUtils.DataSource(inputTestFSSFilePath);
        Instances dataTest = sourceTest.getDataSet();

        // Establecer la clase objetivo (última columna) en el conjunto de datos de prueba
        if (dataTest.classIndex() == -1) {
            dataTest.setClassIndex(dataTest.numAttributes() - 1);
        }

        // Preparar el archivo de salida
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath));
        writer.write("Instantzia,Iragarpena\n");

        // Realizar predicciones para cada instancia en el conjunto de prueba
        for (int i = 0; i < dataTest.numInstances(); i++) {
            // Obtener la instancia a predecir
            double classLabel = svm.classifyInstance(dataTest.instance(i));

            // Convertir el número de clase a su nombre
            String className = dataTest.classAttribute().value((int) classLabel);

            // Escribir el ID de la instancia y la clase predicha en el archivo de salida
            writer.write(i + "," + className + "\n");
        }

        // Cerrar el archivo de salida
        writer.close();
        System.out.println("Iragarpenak gorde dira hurrengo helbidean: " + outputFilePath);
    }
}
