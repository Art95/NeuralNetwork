package main;

import neuralnetwork.NeuralNetwork;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Scanner;

/**
 * Created by Artem on 12.05.2016.
 */
public class Main {
    private static List<List<Double>> inputs;
    private static List<List<Double>> ans;

    public static void main(String[] args) {
        List<Integer> neurons = new ArrayList<>();
        neurons.add(4);
        neurons.add(2);
        neurons.add(3);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neurons);

        readData(".\\Data\\Iris.txt");

        neuralNetwork.trainNetwork(inputs, ans, 0.3);
        neuralNetwork.testNetwork(inputs, ans);
    }

    private static void readData(String fileAddress) {
        inputs = new ArrayList<>();
        ans = new ArrayList<>();


        try {
            File file = new File(fileAddress);
            InputStream inp = new FileInputStream(file);
            Scanner scan = new Scanner(inp);

            while (scan.hasNextLine()) {
                String[] vals = scan.nextLine().split(" ");

                List<Double> input = new ArrayList<>();
                List<Double> answer = new ArrayList<>();
                answer.add(0.0);
                answer.add(0.0);
                answer.add(0.0);

                for (int i = 0; i < 4; ++i) {
                    input.add(Double.parseDouble(vals[i].trim()));
                }

                answer.set(Integer.parseInt(vals[4].trim()), 1.0);

                inputs.add(input);
                ans.add(answer);
            }

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
