import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;

import java.util.ArrayList;

public class ApplyFssToDev {
    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Mesedez, atributua ondo sartu.");
            return;
        }

        String inputDevBowPath = args[0];
        String outputDevBowFssPath = args[1];
        String inputTrainFSSPath = args[2];

        try {
            // Cargar train y dev
            ConverterUtils.DataSource trainSource = new ConverterUtils.DataSource(inputTrainFSSPath);
            Instances trainData = trainSource.getDataSet();
            ConverterUtils.DataSource devSource = new ConverterUtils.DataSource(inputDevBowPath);
            Instances devData = devSource.getDataSet();

            // Asegurar que la clase esté definida
            if (trainData.classIndex() == -1) trainData.setClassIndex(trainData.numAttributes() - 1);
            if (devData.classIndex() == -1) devData.setClassIndex(devData.numAttributes() - 1);

            // Ajustar dev eliminando atributos extra y ordenando
            devData = adjustTestInstancesWithTrainFSS(devData, trainData);

            // Guardar nuevo conjunto de datos
            weka.core.converters.ArffSaver saver = new weka.core.converters.ArffSaver();
            saver.setInstances(devData);
            saver.setFile(new java.io.File(outputDevBowFssPath));
            saver.writeBatch();

            System.out.println("🚀 Conjunto de desarrollo con FSS aplicado guardado en " + outputDevBowFssPath);
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR");
        }
    }

    private static Instances adjustTestInstancesWithTrainFSS(Instances test, Instances train) throws Exception {
        // 🔹 Paso 1: Eliminar atributos que están en test pero no en train
        ArrayList<Integer> indicesToRemove = new ArrayList<>();
        for (int i = 0; i < test.numAttributes(); i++) {
            if (train.attribute(test.attribute(i).name()) == null) {
                indicesToRemove.add(i);
            }
        }

        if (!indicesToRemove.isEmpty()) {
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndicesArray(indicesToRemove.stream().mapToInt(i -> i).toArray());
            removeFilter.setInvertSelection(false);
            removeFilter.setInputFormat(test);
            test = Filter.useFilter(test, removeFilter);
        }

        // 🔹 Paso 2: Reordenar atributos de test para coincidir con train
        StringBuilder order = new StringBuilder();
        for (int i = 0; i < train.numAttributes(); i++) {
            int idx = test.attribute(train.attribute(i).name()).index() + 1;
            order.append(idx).append(",");
        }
        order.deleteCharAt(order.length() - 1); // Quitar la última coma

        Reorder reorderFilter = new Reorder();
        reorderFilter.setAttributeIndices(order.toString());
        reorderFilter.setInputFormat(test);
        test = Filter.useFilter(test, reorderFilter);

        return test;
    }
}
