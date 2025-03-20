import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import java.io.File;

public class getDev {
    public static void main(String[] args) throws Exception {
        // Archivo de entrada y salida
        String inputPath = "data/train_RAW.arff";  // Dataset original
        String trainPath = "data/train_split_RAW.arff"; // Archivo de entrenamiento
        String devPath = "data/dev_RAW.arff"; // Archivo de validación

        // Cargar dataset
        DataSource source = new DataSource(inputPath);
        Instances data = source.getDataSet();

        // Establecer el índice de la clase (último atributo)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Crear filtro para dividir de forma estratificada (80% train - 20% dev)
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
        filter.setNumFolds(5); // 5-fold, 1 fold será dev (20%)
        filter.setFold(1); // Tomamos el primer fold como dev
        filter.setInvertSelection(false); // Filtramos para obtener solo dev
        filter.setInputFormat(data);
        Instances devData = weka.filters.Filter.useFilter(data, filter);

        filter.setInvertSelection(true); // Ahora seleccionamos el 80% restante como train
        Instances trainData = weka.filters.Filter.useFilter(data, filter);

        // Reemplazar la clase en `dev.arff` con `?`
        for (int i = 0; i < devData.numInstances(); i++) {
            devData.instance(i).setMissing(devData.classIndex());
        }

        // Guardar train_split.arff
        ArffSaver trainSaver = new ArffSaver();
        trainSaver.setInstances(trainData);
        trainSaver.setFile(new File(trainPath));
        trainSaver.writeBatch();

        // Guardar dev.arff
        ArffSaver devSaver = new ArffSaver();
        devSaver.setInstances(devData);
        devSaver.setFile(new File(devPath));
        devSaver.writeBatch();

        System.out.println("División completada: Train → " + trainData.numInstances() +
                " instancias, Dev → " + devData.numInstances());
    }
}
