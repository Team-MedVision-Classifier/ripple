package com.ripple.cellpose;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import org.json.JSONObject;

class BackendRunner {

    final Path backendDir;
    final Path modelsDir;

    BackendRunner() {
        this.backendDir = findBackendDirectory();
        if (backendDir != null) {
            this.modelsDir = backendDir.resolve("cellpose backend").resolve("models");
            System.out.println("[Cellpose] Backend directory found: " + backendDir);
            System.out.println("[Cellpose] Models directory: " + modelsDir);
        } else {
            this.modelsDir = null;
            System.err.println("[Cellpose] WARNING: Backend directory not found!");
            System.err.println("[Cellpose] Working directory: " + System.getProperty("user.dir"));
        }
    }

    static Path findBackendDirectory() {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path candidate = projectRoot.resolve("cellpose backend");
        if (candidate.toFile().exists() && candidate.toFile().isDirectory()) {
            return candidate;
        }
        return null;
    }

    String getPythonExecutable(String modelType) {
        if (backendDir == null) {
            return null;
        }
        String venvName = "Cellpose3.1".equals(modelType) ? "venv_v3" : "venv_v4";
        Path pythonPath = backendDir.resolve("cellpose backend")
                                    .resolve(venvName)
                                    .resolve("Scripts")
                                    .resolve("python.exe");
        return pythonPath.toFile().exists() ? pythonPath.toString() : "python";
    }

    String resolvePretrainedModelArg(String modelType, String modelName) throws Exception {
        if ("cyto3".equals(modelName) || "cpsam".equals(modelName)) {
            return modelName;
        }
        if (modelsDir == null) {
            throw new Exception("Models directory not found.");
        }
        Path modelSubdir = "Cellpose3.1".equals(modelType)
            ? modelsDir.resolve("Cellpose 3.1")
            : modelsDir.resolve("CellposeSAM");
        Path modelPath = modelSubdir.resolve(modelName).toAbsolutePath();
        if (!modelPath.toFile().exists()) {
            throw new Exception("Selected model file not found: " + modelPath);
        }
        return modelPath.toString();
    }

    static int[] parseChannels(String channelsText) {
        int chan = 0;
        int chan2 = 0;
        if (channelsText != null) {
            String[] parts = channelsText.split(",");
            if (parts.length > 0) {
                chan = parseChannelIndex(parts[0], 0);
            }
            if (parts.length > 1) {
                chan2 = parseChannelIndex(parts[1], 0);
            }
        }
        return new int[]{chan, chan2};
    }

    static int parseChannelIndex(String value, int defaultValue) {
        try {
            int parsed = Integer.parseInt(value.trim());
            return Math.max(0, Math.min(3, parsed));
        } catch (Exception ignored) {
            return defaultValue;
        }
    }

    static boolean hasNoMasksMessage(String processLog) {
        return processLog != null && processLog.toLowerCase().contains("no masks found");
    }

    static File findMaskOutput(File outputDir, File inputFile) {
        String baseName = OutputPaths.baseName(inputFile);
        File expected = new File(outputDir, baseName + "_cp_masks.png");
        if (expected.exists()) {
            return expected;
        }
        File[] candidates = outputDir.listFiles((dir, name) -> name.endsWith("_cp_masks.png"));
        return (candidates != null && candidates.length > 0) ? candidates[0] : null;
    }

    SegmentationResult segmentImageFile(
        File imageFile,
        String modelType,
        String pretrainedModelArg,
        String diameter,
        String flowThreshold,
        String cellprobThreshold,
        boolean useGpu,
        int[] parsedChannels
    ) throws Exception {
        if (backendDir == null) {
            throw new Exception("Python executable not found for " + modelType);
        }

        File outputDir = new File(System.getProperty("java.io.tmpdir"), "cellpose_out_" + System.nanoTime());
        if (!outputDir.mkdirs() && !outputDir.exists()) {
            throw new Exception("Failed to create temporary output directory: " + outputDir);
        }

        try {
            String pythonExe = getPythonExecutable(modelType);
            List<String> command = new ArrayList<>();
            command.add(pythonExe);
            command.add("-m");
            command.add("cellpose");
            command.add("--image_path");
            command.add(imageFile.getAbsolutePath());
            command.add("--pretrained_model");
            command.add(pretrainedModelArg);
            command.add("--diameter");
            command.add(diameter);
            command.add("--flow_threshold");
            command.add(flowThreshold);
            command.add("--cellprob_threshold");
            command.add(cellprobThreshold);
            command.add("--save_png");
            command.add("--savedir");
            command.add(outputDir.getAbsolutePath());
            command.add("--no_npy");

            if ("Cellpose3.1".equals(modelType)) {
                command.add("--chan");
                command.add(String.valueOf(parsedChannels[0]));
                command.add("--chan2");
                command.add(String.valueOf(parsedChannels[1]));
            }
            if (useGpu) {
                command.add("--use_gpu");
            }

            System.out.println("[Cellpose CLI] Executing command:");
            System.out.println("[Cellpose CLI] " + String.join(" ", command));

            ProcessBuilder pb = new ProcessBuilder(command);
            pb.directory(backendDir.resolve("cellpose backend").toFile());
            pb.redirectErrorStream(true);
            System.out.println("[Cellpose CLI] Working directory: " + pb.directory());

            Process process = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder log = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                log.append(line).append("\n");
                System.out.println("[Cellpose CLI] " + line);
            }
            reader.close();

            int exitCode = process.waitFor();
            System.out.println("[Cellpose CLI] Process exited with code: " + exitCode);

            if (exitCode != 0) {
                throw new Exception("Segmentation failed for " + imageFile.getName()
                    + " with exit code " + exitCode + "\n\nCLI output:\n" + log);
            }

            File maskFile = findMaskOutput(outputDir, imageFile);
            if (maskFile == null || !maskFile.exists()) {
                if (hasNoMasksMessage(log.toString())) {
                    return new SegmentationResult(null, true);
                }
                throw new Exception(
                    "Segmentation completed but mask file was not found for " + imageFile.getName()
                    + " in: " + outputDir + "\nExpected suffix: _cp_masks.png\n\nCLI output:\n" + log);
            }

            BufferedImage rawMask = ImageIO.read(maskFile);
            if (rawMask == null) {
                throw new Exception("Could not read generated mask file: " + maskFile);
            }
            return new SegmentationResult(rawMask, false);
        } finally {
            deleteRecursively(outputDir);
        }
    }

    JSONObject executeDinoClassification(
        File imageFile, File maskFile, File modelFile, int fociChannel, boolean useGpu
    ) throws Exception {
        if (backendDir == null) {
            return new JSONObject().put("error", "Python executable not found");
        }

        String pythonExe = getPythonExecutable("CellposeSAM");
        if (pythonExe == null) {
            return new JSONObject().put("error", "Python executable not found");
        }

        Path scriptPath = backendDir.resolve("cellpose backend").resolve("classify_foci.py");
        if (!scriptPath.toFile().exists()) {
            return new JSONObject().put("error", "classify_foci.py not found at: " + scriptPath);
        }

        List<String> command = new ArrayList<>();
        command.add(pythonExe);
        command.add(scriptPath.toString());
        command.add("--image_path");
        command.add(imageFile.getAbsolutePath());
        command.add("--mask_path");
        command.add(maskFile.getAbsolutePath());
        command.add("--model_path");
        command.add(modelFile.getAbsolutePath());
        command.add("--image_size");
        command.add("384");
        command.add("--backbone_name");
        command.add("vit_tiny_patch16_384");
        command.add("--foci_channel");
        command.add(String.valueOf(fociChannel));
        if (useGpu) {
            command.add("--use_gpu");
        }

        System.out.println("[DINO Classify] Executing: " + String.join(" ", command));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(backendDir.resolve("cellpose backend").toFile());
        pb.redirectErrorStream(false);
        Process process = pb.start();

        BufferedReader stdoutReader = new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
        StringBuilder jsonOutput = new StringBuilder();
        String line;
        while ((line = stdoutReader.readLine()) != null) {
            jsonOutput.append(line);
        }
        stdoutReader.close();

        BufferedReader stderrReader = new BufferedReader(
            new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8));
        StringBuilder stderrOutput = new StringBuilder();
        while ((line = stderrReader.readLine()) != null) {
            stderrOutput.append(line).append("\n");
            System.out.println("[DINO Classify stderr] " + line);
        }
        stderrReader.close();

        int exitCode = process.waitFor();
        System.out.println("[DINO Classify] Exit code: " + exitCode);

        if (jsonOutput.length() == 0) {
            String errorMsg = stderrOutput.length() > 0
                ? stderrOutput.toString().trim()
                : "No output from classification script (exit code " + exitCode + ")";
            return new JSONObject().put("error", errorMsg);
        }
        try {
            return new JSONObject(jsonOutput.toString());
        } catch (Exception e) {
            return new JSONObject().put("error",
                "Invalid JSON output: " + jsonOutput.toString().substring(0, Math.min(200, jsonOutput.length())));
        }
    }

    JSONObject executeThresholdClassification(
        File imageFile, File maskFile, int fociChannel,
        int denoiseH, int tophatSize, double clipLimit,
        int minArea, int damageThreshold
    ) throws Exception {
        if (backendDir == null) {
            return new JSONObject().put("error", "Python executable not found");
        }

        String pythonExe = getPythonExecutable("Cellpose3.1");
        if (pythonExe == null) {
            pythonExe = getPythonExecutable("CellposeSAM");
        }
        if (pythonExe == null) {
            return new JSONObject().put("error", "Python executable not found");
        }

        Path scriptPath = backendDir.resolve("cellpose backend").resolve("classify_foci_threshold.py");
        if (!scriptPath.toFile().exists()) {
            return new JSONObject().put("error", "classify_foci_threshold.py not found at: " + scriptPath);
        }

        List<String> command = new ArrayList<>();
        command.add(pythonExe);
        command.add(scriptPath.toString());
        command.add("--image_path");
        command.add(imageFile.getAbsolutePath());
        command.add("--mask_path");
        command.add(maskFile.getAbsolutePath());
        command.add("--foci_channel");
        command.add(String.valueOf(fociChannel));
        command.add("--denoise_h");
        command.add(String.valueOf(denoiseH));
        command.add("--tophat_size");
        command.add(String.valueOf(tophatSize));
        command.add("--clip_limit");
        command.add(String.valueOf(clipLimit));
        command.add("--min_area");
        command.add(String.valueOf(minArea));
        command.add("--damage_threshold");
        command.add(String.valueOf(damageThreshold));
        command.add("--save_foci_mask");

        System.out.println("[Threshold Classify] Executing: " + String.join(" ", command));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(backendDir.resolve("cellpose backend").toFile());
        pb.redirectErrorStream(false);
        Process process = pb.start();

        BufferedReader stdoutReader = new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
        StringBuilder jsonOutput = new StringBuilder();
        String line;
        while ((line = stdoutReader.readLine()) != null) {
            jsonOutput.append(line);
        }
        stdoutReader.close();

        BufferedReader stderrReader = new BufferedReader(
            new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8));
        StringBuilder stderrOutput = new StringBuilder();
        while ((line = stderrReader.readLine()) != null) {
            stderrOutput.append(line).append("\n");
            System.out.println("[Threshold Classify stderr] " + line);
        }
        stderrReader.close();

        int exitCode = process.waitFor();
        System.out.println("[Threshold Classify] Exit code: " + exitCode);

        if (jsonOutput.length() == 0) {
            String errorMsg = stderrOutput.length() > 0
                ? stderrOutput.toString().trim()
                : "No output from threshold script (exit code " + exitCode + ")";
            return new JSONObject().put("error", errorMsg);
        }
        try {
            return new JSONObject(jsonOutput.toString());
        } catch (Exception e) {
            return new JSONObject().put("error",
                "Invalid JSON output: " + jsonOutput.toString().substring(0, Math.min(200, jsonOutput.length())));
        }
    }

    static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursively(child);
                }
            }
        }
        if (!file.delete()) {
            file.deleteOnExit();
        }
    }
}
