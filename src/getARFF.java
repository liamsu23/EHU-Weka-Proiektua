import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class getARFF {
    public static void main(String[] args) throws IOException {
        // Verificar argumentos
        if (args.length != 4) {
            System.out.println("Uso: java getArff2 <train_csv> <test_csv> <train_arff> <test_arff>");
            return;
        }

        String trainCSVPath = args[0];
        String testCSVPath = args[1];
        String trainARFFPath = args[2];
        String testARFFPath = args[3];

        // Cargar el mapeo de etiquetas normalizado
        Map<String, String> labelMapping = loadLabelMapping();

        try {
            // Convertir archivos
            convertCSVtoARFF(trainCSVPath, trainARFFPath, labelMapping, false);
            convertCSVtoARFF(testCSVPath, testARFFPath, labelMapping, true);

            System.out.println("Archivos ARFF generados correctamente.");
        } catch (IOException | CsvException e) {
            e.printStackTrace();
            System.out.println("ERROR: " + e.getMessage());
        }
    }

    private static void convertCSVtoARFF(String csvFilePath, String arffFilePath,
                                         Map<String, String> labelMapping, boolean isTestData)
            throws IOException, CsvException {

        CSVReader reader = new CSVReader(new FileReader(csvFilePath));
        List<String[]> csvData = reader.readAll();

        Set<String> places = new HashSet<>();
        Set<String> generalCategories = new HashSet<>();

        // Procesar datos para obtener valores únicos
        for (int i = 1; i < csvData.size(); i++) {
            places.add(csvData.get(i)[4]); // Columna Place

            // Para categorías generales (solo en train o usando mapeo existente)
            String specificLabel = csvData.get(i)[6].trim().toLowerCase();
            String generalLabel = labelMapping.getOrDefault(specificLabel, specificLabel);
            generalLabel = generalLabel.replaceAll("[^a-zA-Z0-9]", "_");
            generalCategories.add(generalLabel);
        }

        // Escribir archivo ARFF
        FileWriter arffWriter = new FileWriter(arffFilePath);
        writeHeaderARFF(arffWriter, places, generalCategories);

        // Procesar y escribir cada fila
        for (int i = 1; i < csvData.size(); i++) {
            String[] row = csvData.get(i).clone(); // Copiar para no modificar el original

            // Procesar Cause_of_Death (columna 6)
            String specificLabel = row[6].trim().toLowerCase();
            String generalLabel = labelMapping.getOrDefault(specificLabel, specificLabel);
            generalLabel = generalLabel.replaceAll("[^a-zA-Z0-9?]", "_");

            // Asignar valor final
            if (isTestData) {
                row[6] = "?";
            } else {
                row[6] = specificLabel.equalsIgnoreCase("na") ? "?" : generalLabel;
            }

            // Limpiar narrative (columna 5)
            row[5] = "\"" + cleanNarrative(row[5]) + "\"";

            arffWriter.write(String.join(",", row) + "\n");
        }

        arffWriter.close();
        reader.close();
    }

    private static Map<String, String> loadLabelMapping() {
        Map<String, String> labelMapping = new HashMap<>();

        // Enfermedades infecciosas (CIE-10: A00-B99)
        labelMapping.put("diarrhea/dysentery", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("other infectious diseases", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("aids", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("sepsis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("meningitis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("meningitis/sepsis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("malaria", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("encephalitis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("measles", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("hemorrhagic fever", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("tb", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("hepatitis", "Certain_infectious_and_Parasitic_Diseases");
        labelMapping.put("tetanus", "Certain_infectious_and_Parasitic_Diseases");

        // Tumores (CIE-10: C00-D49)
        labelMapping.put("leukemia/lymphomas", "Neoplasms");
        labelMapping.put("colorectal cancer", "Neoplasms");
        labelMapping.put("lung cancer", "Neoplasms");
        labelMapping.put("cervical cancer", "Neoplasms");
        labelMapping.put("breast cancer", "Neoplasms");
        labelMapping.put("stomach cancer", "Neoplasms");
        labelMapping.put("prostate cancer", "Neoplasms");
        labelMapping.put("esophageal cancer", "Neoplasms");
        labelMapping.put("liver cancer", "Neoplasms");
        labelMapping.put("pancreatic cancer", "Neoplasms");
        labelMapping.put("other cancers", "Neoplasms");

        // Enfermedades endocrinas (CIE-10: E00-E90)
        labelMapping.put("diabetes", "Endocrine_Nutritional_and_Metabolic_Diseases");
        labelMapping.put("malnutrition", "Endocrine_Nutritional_and_Metabolic_Diseases");
        labelMapping.put("obesity", "Endocrine_Nutritional_and_Metabolic_Diseases");

        // Enfermedades neurológicas (CIE-10: G00-G99)
        labelMapping.put("epilepsy", "Diseases_of_the_Nervous_System");
        labelMapping.put("alzheimer", "Diseases_of_the_Nervous_System");
        labelMapping.put("parkinson", "Diseases_of_the_Nervous_System");

        // Enfermedades cardiovasculares (CIE-10: I00-I99)
        labelMapping.put("stroke", "Diseases_of_the_circulatory_system");
        labelMapping.put("acute myocardial infarction", "Diseases_of_the_circulatory_system");
        labelMapping.put("heart failure", "Diseases_of_the_circulatory_system");
        labelMapping.put("hypertension", "Diseases_of_the_circulatory_system");
        labelMapping.put("cardiac arrest", "Diseases_of_the_circulatory_system");

        // Enfermedades respiratorias (CIE-10: J00-J99)
        labelMapping.put("pneumonia", "Diseases_of_Respiratory_System");
        labelMapping.put("asthma", "Diseases_of_Respiratory_System");
        labelMapping.put("copd", "Diseases_of_Respiratory_System");
        labelMapping.put("tuberculosis respiratory", "Diseases_of_Respiratory_System");

        // Enfermedades digestivas (CIE-10: K00-K95)
        labelMapping.put("cirrhosis", "Diseases_of_the_Digestive_System");
        labelMapping.put("other digestive diseases", "Diseases_of_the_Digestive_System");
        labelMapping.put("gastritis", "Diseases_of_the_Digestive_System");
        labelMapping.put("peptic ulcer", "Diseases_of_the_Digestive_System");

        // Enfermedades genitourinarias (CIE-10: N00-N99)
        labelMapping.put("renal failure", "Diseases_of_the_Genitourinary_System");
        labelMapping.put("kidney disease", "Diseases_of_the_Genitourinary_System");

        // Embarazo/parto (CIE-10: O00-O9A)
        labelMapping.put("preterm delivery", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("stillbirth", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("maternal", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("birth asphyxia", "Pregnancy_childbirth_and_the_puerperium");
        labelMapping.put("postpartum hemorrhage", "Pregnancy_childbirth_and_the_puerperium");

        // Malformaciones congénitas (CIE-10: Q00-Q99)
        labelMapping.put("congenital malformations", "Congenital_Malformations");

        // Causas externas (CIE-10: V01-Y99)
        labelMapping.put("bite of venomous animal", "Injury_Poisoning_and_External_Causes");
        labelMapping.put("poisonings", "Injury_Poisoning_and_External_Causes");
        labelMapping.put("road traffic", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("falls", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("homicide", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("fires", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("drowning", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("suicide", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("violent death", "External_Causes_of_Morbidity_and_Mortality");
        labelMapping.put("other injuries", "External_Causes_of_Morbidity_and_Mortality");

        return labelMapping;
    }
    private static String cleanNarrative(String text) {
        return text.replaceAll("[#?!,\"'.;:()\\[\\]{}/\\\\]", " ").trim();
    }

    private static void writeHeaderARFF(FileWriter fw, Set<String> places,
                                        Set<String> generalCategories) throws IOException {
        fw.write("@RELATION muertes_causas\n\n");
        fw.write("@ATTRIBUTE ID NUMERIC\n");
        fw.write("@ATTRIBUTE Module {Adult, Child, Neonate}\n");
        fw.write("@ATTRIBUTE Age NUMERIC\n");
        fw.write("@ATTRIBUTE Sex {1, 2}\n");
        fw.write("@ATTRIBUTE Site {" + String.join(",", places) + "}\n");
        fw.write("@ATTRIBUTE Open_Response STRING\n");
        fw.write("@ATTRIBUTE Cause_of_Death {" + String.join(",", generalCategories) + "}\n\n");
        fw.write("@DATA\n");
    }
}