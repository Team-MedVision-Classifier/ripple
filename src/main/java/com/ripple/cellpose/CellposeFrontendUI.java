package com.ripple.cellpose;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import javax.imageio.ImageIO;
import org.json.JSONArray;
import org.json.JSONObject;
import ij.ImagePlus;
import ij.io.Opener;

/**
 * Standalone Cellpose Frontend (No Fiji/ImageJ Plugin Dependencies)
 * Runs Cellpose Backend models directly without requiring a separate server.
 */
public class CellposeFrontendUI {

    private static class MaskPropertiesAccumulator {
        final int label;
        int area;
        long sumX;
        long sumY;
        double sumXX;
        double sumYY;
        double sumXY;
        int minX = Integer.MAX_VALUE;
        int minY = Integer.MAX_VALUE;
        int maxX = Integer.MIN_VALUE;
        int maxY = Integer.MIN_VALUE;
        int boundaryPixels;

        MaskPropertiesAccumulator(int label) {
            this.label = label;
        }

        void addPixel(int x, int y) {
            area++;
            sumX += x;
            sumY += y;
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }

        double centroidX() {
            return area > 0 ? (double) sumX / area : 0.0;
        }

        double centroidY() {
            return area > 0 ? (double) sumY / area : 0.0;
        }

        double equivalentDiameter() {
            return area > 0 ? Math.sqrt(4.0 * area / Math.PI) : 0.0;
        }

        /** Eccentricity from the inertia tensor eigenvalues. Returns 0 for a circle. */
        double eccentricity() {
            if (area <= 1) return 0.0;
            double cx = centroidX();
            double cy = centroidY();
            double mu20 = sumXX / area - cx * cx;
            double mu02 = sumYY / area - cy * cy;
            double mu11 = sumXY / area - cx * cy;
            double diff = mu20 - mu02;
            double root = Math.sqrt(diff * diff + 4.0 * mu11 * mu11);
            double lambda1 = (mu20 + mu02 + root) / 2.0;
            double lambda2 = (mu20 + mu02 - root) / 2.0;
            if (lambda1 <= 0.0) return 0.0;
            return Math.sqrt(1.0 - lambda2 / lambda1);
        }
    }

    private static class ImageCellStats {
        final String imageName;
        final int cellCount;
        final double avgArea;
        final double minArea;
        final double maxArea;
        final double avgDiameter;
        final double minDiameter;
        final double maxDiameter;
        final double avgEccentricity;

        ImageCellStats(String imageName, int cellCount,
                       double avgArea, double minArea, double maxArea,
                       double avgDiameter, double minDiameter, double maxDiameter,
                       double avgEccentricity) {
            this.imageName = imageName;
            this.cellCount = cellCount;
            this.avgArea = avgArea;
            this.minArea = minArea;
            this.maxArea = maxArea;
            this.avgDiameter = avgDiameter;
            this.minDiameter = minDiameter;
            this.maxDiameter = maxDiameter;
            this.avgEccentricity = avgEccentricity;
        }
    }

    private static final int LEFT_PANEL_WIDTH = 360;
    private static final int LEFT_PANEL_MIN_WIDTH = 360;

    private JFrame frame;
    private JLabel imageLabel;
    private JLabel statusLabel;
    private BufferedImage originalImage;
    private BufferedImage displayImage;
    private BufferedImage maskImage;
    private ImagePlus imagePlus;
    private File currentImageFile;
    private File currentFolder;
    private JSplitPane fileBrowserSplitPane;
    private JScrollPane imageScrollPane;
    private JScrollPane folderListScrollPane;
    private JPanel emptyFolderPanel;
    private JList<File> folderFileList;
    private DefaultListModel<File> folderFileListModel;

    // Display settings
    private double zoomFactor = 1.0;
    private boolean showMask = true;
    private float maskOpacity = 0.5f;

    // Backend
    private final BackendRunner backendRunner;

    // UI Components
    private JComboBox<String> modelTypeCombo;
    private JComboBox<String> modelNameCombo;
    private JSpinner diameterSpinner;
    private JSpinner flowThresholdSpinner;
    private JSpinner cellprobThresholdSpinner;
    private JTextField channelsField;
    private JCheckBox useGpuCheckbox;
    private JCheckBox showMaskCheckbox;
    private JSlider opacitySlider;
    private JButton segmentButton;
    private JButton segmentAllButton;
    private JLabel segmentationSummaryLabel;
    private JProgressBar progressBar;

    // Color map for mask visualization
    private Color[] colorMap;

    // Classification components — shared
    private JComboBox<String> classificationMethodCombo;
    private JSpinner fociChannelSpinner;
    private JButton classifyButton;
    private JButton classifyAllButton;
    private JCheckBox showClassificationCheckbox;
    private JCheckBox showLabelsCheckbox;
    private JCheckBox showFociMaskCheckbox;
    private JLabel classificationSummaryLabel;
    private BufferedImage classificationOverlay;
    private BufferedImage fociMaskImage;
    private boolean showClassification = true;
    private boolean showLabels = false;
    private boolean showFociMask = false;
    private JPanel methodOptionsCards;
    private CardLayout methodOptionsLayout;

    // DINO-specific
    private JComboBox<String> classificationModelCombo;

    // Threshold-specific
    private JSpinner denoiseHSpinner;
    private JSpinner tophatSizeSpinner;
    private JSpinner clipLimitSpinner;
    private JSpinner minAreaSpinner;
    private JSpinner damageThresholdSpinner;

    // Maps cell label -> classification result: {prediction, probability, bbox}
    private Map<Integer, JSONObject> classificationResults = new HashMap<>();
    // Maps cell label -> centroid (x,y) from segmentation mask
    private Map<Integer, int[]> cellCentroids = new HashMap<>();

    public CellposeFrontendUI(String backendUrl) {
        this.backendRunner = new BackendRunner();
        initializeColorMap();
    }

    public void initUI() {
        frame = new JFrame("Cellpose - Cell Segmentation");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(1400, 900);
        frame.setLayout(new BorderLayout());

        createMenuBar();

        JSplitPane mainSplit = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        mainSplit.setDividerLocation(LEFT_PANEL_WIDTH);
        mainSplit.setResizeWeight(0.0);

        JPanel controlPanel = createControlPanel();
        JScrollPane controlScroll = new JScrollPane(controlPanel);
        controlScroll.setPreferredSize(new Dimension(LEFT_PANEL_WIDTH, 0));
        controlScroll.setMinimumSize(new Dimension(LEFT_PANEL_MIN_WIDTH, 0));
        controlScroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        controlScroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        mainSplit.setLeftComponent(controlScroll);

        JPanel displayPanel = createDisplayPanel();
        mainSplit.setRightComponent(displayPanel);

        frame.add(mainSplit, BorderLayout.CENTER);

        statusLabel = new JLabel(" Ready - Load an image to start segmentation");
        statusLabel.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 10));
        statusLabel.setFont(new Font("Arial", Font.PLAIN, 12));
        frame.add(statusLabel, BorderLayout.SOUTH);

        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        loadModels();
        loadClassificationModels();
    }

    private void createMenuBar() {
        JMenuBar menuBar = new JMenuBar();

        JMenu fileMenu = new JMenu("File");
        JMenuItem openItem = new JMenuItem("Open Image...");
        openItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_O, InputEvent.CTRL_DOWN_MASK));
        openItem.addActionListener(e -> openImage());
        fileMenu.add(openItem);

        JMenuItem openFolderItem = new JMenuItem("Open Folder...");
        openFolderItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_O, InputEvent.CTRL_DOWN_MASK | InputEvent.SHIFT_DOWN_MASK));
        openFolderItem.addActionListener(e -> openFolder());
        fileMenu.add(openFolderItem);

        fileMenu.addSeparator();

        JMenuItem exportMaskItem = new JMenuItem("Export Mask...");
        exportMaskItem.addActionListener(e -> exportMask());
        fileMenu.add(exportMaskItem);

        JMenuItem exportOverlayItem = new JMenuItem("Export Overlay...");
        exportOverlayItem.addActionListener(e -> exportOverlay());
        fileMenu.add(exportOverlayItem);

        fileMenu.addSeparator();

        JMenuItem closeItem = new JMenuItem("Close");
        closeItem.addActionListener(e -> frame.dispose());
        fileMenu.add(closeItem);

        JMenu viewMenu = new JMenu("View");
        JMenuItem zoomInItem = new JMenuItem("Zoom In");
        zoomInItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_PLUS, InputEvent.CTRL_DOWN_MASK));
        zoomInItem.addActionListener(e -> zoomIn());
        viewMenu.add(zoomInItem);

        JMenuItem zoomOutItem = new JMenuItem("Zoom Out");
        zoomOutItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_MINUS, InputEvent.CTRL_DOWN_MASK));
        zoomOutItem.addActionListener(e -> zoomOut());
        viewMenu.add(zoomOutItem);

        JMenuItem zoomFitItem = new JMenuItem("Fit to Window");
        zoomFitItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_0, InputEvent.CTRL_DOWN_MASK));
        zoomFitItem.addActionListener(e -> zoomFit());
        viewMenu.add(zoomFitItem);

        menuBar.add(fileMenu);
        menuBar.add(viewMenu);

        frame.setJMenuBar(menuBar);
    }

    private JPanel createControlPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel modelPanel = createSection("Model Selection");
        modelTypeCombo = new JComboBox<>(new String[]{"CellposeSAM", "Cellpose3.1"});
        modelTypeCombo.addActionListener(e -> {
            updateChannelSelectionState();
            loadModels();
        });
        modelNameCombo = new JComboBox<>();

        addLabeledComponent(modelPanel, "Model Type:", modelTypeCombo);
        addLabeledComponent(modelPanel, "Model:", modelNameCombo);

        JButton refreshBtn = new JButton("Refresh Models");
        refreshBtn.addActionListener(e -> loadModels());
        refreshBtn.setAlignmentX(Component.LEFT_ALIGNMENT);
        modelPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        modelPanel.add(refreshBtn);

        panel.add(modelPanel);
        panel.add(Box.createRigidArea(new Dimension(0, 10)));

        JPanel paramPanel = createSection("Segmentation Parameters");

        diameterSpinner = new JSpinner(new SpinnerNumberModel(30.0, 0.0, 500.0, 5.0));
        flowThresholdSpinner = new JSpinner(new SpinnerNumberModel(0.4, 0.0, 3.0, 0.1));
        cellprobThresholdSpinner = new JSpinner(new SpinnerNumberModel(0.0, -6.0, 6.0, 0.5));
        channelsField = new JTextField("0,0", 10);
        useGpuCheckbox = new JCheckBox("Use GPU");

        addLabeledComponent(paramPanel, "Cell Diameter (px):", diameterSpinner);
        addLabeledComponent(paramPanel, "Flow Threshold:", flowThresholdSpinner);
        addLabeledComponent(paramPanel, "Cell Prob Threshold:", cellprobThresholdSpinner);
        addLabeledComponent(paramPanel, "Channels:", channelsField);
        paramPanel.add(useGpuCheckbox);

        // Channel flags are only used by Cellpose 3.1 in CLI mode.
        updateChannelSelectionState();

        panel.add(paramPanel);
        panel.add(Box.createRigidArea(new Dimension(0, 10)));

        JPanel actionPanel = createSection("Actions");

        JPanel segButtonRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
        segButtonRow.setAlignmentX(Component.LEFT_ALIGNMENT);
        segmentButton = new JButton("Run Segmentation");
        segmentButton.setFont(new Font("Arial", Font.BOLD, 13));
        segmentButton.addActionListener(e -> runSegmentation());
        segmentAllButton = new JButton("Segment All");
        segmentAllButton.setFont(new Font("Arial", Font.BOLD, 13));
        segmentAllButton.setEnabled(false);
        segmentAllButton.addActionListener(e -> runBatchSegmentation());
        segButtonRow.add(segmentButton);
        segButtonRow.add(segmentAllButton);
        actionPanel.add(segButtonRow);

        actionPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        segmentationSummaryLabel = new JLabel("No segmentation results");
        segmentationSummaryLabel.setFont(new Font("Arial", Font.PLAIN, 11));
        segmentationSummaryLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        actionPanel.add(segmentationSummaryLabel);

        actionPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        progressBar = new JProgressBar();
        progressBar.setStringPainted(true);
        progressBar.setString("Ready");
        progressBar.setAlignmentX(Component.LEFT_ALIGNMENT);
        actionPanel.add(progressBar);

        panel.add(actionPanel);
        panel.add(Box.createRigidArea(new Dimension(0, 10)));

        JPanel classificationPanel = createSection("Foci Classification");

        classificationMethodCombo = new JComboBox<>(new String[]{"DINO", "Threshold"});
        addLabeledComponent(classificationPanel, "Method:", classificationMethodCombo);

        fociChannelSpinner = new JSpinner(new SpinnerNumberModel(-1, -1, 20, 1));
        fociChannelSpinner.setToolTipText(
            "Channel containing foci (-1 = auto/default, 0 = first channel, 1 = second channel, ...)");
        addLabeledComponent(classificationPanel, "Foci Channel:", fociChannelSpinner);

        methodOptionsLayout = new CardLayout();
        methodOptionsCards = new JPanel(methodOptionsLayout);
        methodOptionsCards.setAlignmentX(Component.LEFT_ALIGNMENT);

        JPanel dinoOptionsPanel = new JPanel();
        dinoOptionsPanel.setLayout(new BoxLayout(dinoOptionsPanel, BoxLayout.Y_AXIS));
        classificationModelCombo = new JComboBox<>();
        addLabeledComponent(dinoOptionsPanel, "Model:", classificationModelCombo);
        JButton refreshClassModelsBtn = new JButton("Refresh Models");
        refreshClassModelsBtn.addActionListener(e -> loadClassificationModels());
        refreshClassModelsBtn.setAlignmentX(Component.LEFT_ALIGNMENT);
        dinoOptionsPanel.add(refreshClassModelsBtn);
        methodOptionsCards.add(dinoOptionsPanel, "DINO");

        JPanel thresholdOptionsPanel = new JPanel();
        thresholdOptionsPanel.setLayout(new BoxLayout(thresholdOptionsPanel, BoxLayout.Y_AXIS));
        denoiseHSpinner = new JSpinner(new SpinnerNumberModel(15, 0, 50, 1));
        denoiseHSpinner.setToolTipText("NLM Denoising strength (0 = disabled)");
        addLabeledComponent(thresholdOptionsPanel, "Denoise H:", denoiseHSpinner);
        tophatSizeSpinner = new JSpinner(new SpinnerNumberModel(13, 3, 51, 2));
        tophatSizeSpinner.setToolTipText("Top-Hat kernel size (odd, controls max foci size to isolate)");
        addLabeledComponent(thresholdOptionsPanel, "TopHat Size:", tophatSizeSpinner);
        clipLimitSpinner = new JSpinner(new SpinnerNumberModel(6.0, 0.1, 20.0, 0.5));
        clipLimitSpinner.setToolTipText("CLAHE clip limit for contrast enhancement");
        addLabeledComponent(thresholdOptionsPanel, "CLAHE Clip Limit:", clipLimitSpinner);
        minAreaSpinner = new JSpinner(new SpinnerNumberModel(11, 0, 500, 1));
        minAreaSpinner.setToolTipText("Minimum foci area in pixels (smaller spots are removed)");
        addLabeledComponent(thresholdOptionsPanel, "Min Foci Area (px):", minAreaSpinner);
        damageThresholdSpinner = new JSpinner(new SpinnerNumberModel(5, 1, 100, 1));
        damageThresholdSpinner.setToolTipText("If a cell has >= this many foci, it is classified as Damaged");
        addLabeledComponent(thresholdOptionsPanel, "Damage Threshold:", damageThresholdSpinner);
        methodOptionsCards.add(thresholdOptionsPanel, "Threshold");

        classificationPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        classificationPanel.add(methodOptionsCards);

        classificationMethodCombo.addActionListener(e -> {
            String selected = (String) classificationMethodCombo.getSelectedItem();
            if (selected != null) {
                methodOptionsLayout.show(methodOptionsCards, selected);
            }
        });

        classificationPanel.add(Box.createRigidArea(new Dimension(0, 10)));

        JPanel classifyButtonRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
        classifyButtonRow.setAlignmentX(Component.LEFT_ALIGNMENT);
        classifyButton = new JButton("Run Classification");
        classifyButton.setFont(new Font("Arial", Font.BOLD, 13));
        classifyButton.addActionListener(e -> runClassification());
        classifyAllButton = new JButton("Classify All");
        classifyAllButton.setFont(new Font("Arial", Font.BOLD, 13));
        classifyAllButton.setEnabled(false);
        classifyAllButton.addActionListener(e -> runBatchClassification());
        classifyButtonRow.add(classifyButton);
        classifyButtonRow.add(classifyAllButton);
        classificationPanel.add(classifyButtonRow);

        classificationPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        classificationSummaryLabel = new JLabel("No classification results");
        classificationSummaryLabel.setFont(new Font("Arial", Font.PLAIN, 11));
        classificationSummaryLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        classificationPanel.add(classificationSummaryLabel);

        panel.add(classificationPanel);
        panel.add(Box.createRigidArea(new Dimension(0, 10)));

        JPanel displaySettingsPanel = createSection("Display Settings");

        showMaskCheckbox = new JCheckBox("Show Mask Overlay", true);
        showMaskCheckbox.addActionListener(e -> {
            showMask = showMaskCheckbox.isSelected();
            updateDisplay();
        });
        displaySettingsPanel.add(showMaskCheckbox);

        showClassificationCheckbox = new JCheckBox("Show Classification Overlay", true);
        showClassificationCheckbox.addActionListener(e -> {
            showClassification = showClassificationCheckbox.isSelected();
            updateDisplay();
        });
        displaySettingsPanel.add(showClassificationCheckbox);

        showLabelsCheckbox = new JCheckBox("Show Cell Labels", false);
        showLabelsCheckbox.addActionListener(e -> {
            showLabels = showLabelsCheckbox.isSelected();
            updateDisplay();
        });
        displaySettingsPanel.add(showLabelsCheckbox);

        showFociMaskCheckbox = new JCheckBox("Show Foci Mask", false);
        showFociMaskCheckbox.setToolTipText("Show detected foci from threshold method");
        showFociMaskCheckbox.addActionListener(e -> {
            showFociMask = showFociMaskCheckbox.isSelected();
            updateDisplay();
        });
        displaySettingsPanel.add(showFociMaskCheckbox);

        displaySettingsPanel.add(Box.createRigidArea(new Dimension(0, 10)));
        displaySettingsPanel.add(new JLabel("Mask Opacity:"));
        opacitySlider = new JSlider(0, 100, 50);
        opacitySlider.setMajorTickSpacing(25);
        opacitySlider.setPaintTicks(true);
        opacitySlider.setPaintLabels(true);
        opacitySlider.addChangeListener(e -> {
            maskOpacity = opacitySlider.getValue() / 100.0f;
            updateDisplay();
        });
        displaySettingsPanel.add(opacitySlider);

        panel.add(displaySettingsPanel);
        panel.add(Box.createVerticalGlue());

        return panel;
    }

    private JPanel createDisplayPanel() {
        JPanel panel = new JPanel(new BorderLayout());

        imageLabel = new JLabel("No image loaded", SwingConstants.CENTER);
        imageLabel.setFont(new Font("Arial", Font.PLAIN, 16));
        imageLabel.setForeground(Color.GRAY);

        imageScrollPane = new JScrollPane(imageLabel);
        imageScrollPane.setBackground(Color.DARK_GRAY);

        folderFileListModel = new DefaultListModel<>();
        folderFileList = new JList<>(folderFileListModel);
        folderFileList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        folderFileList.setCellRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(
                JList<?> list, Object value, int index, boolean isSelected, boolean cellHasFocus
            ) {
                super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
                if (value instanceof File file) {
                    setText(file.getName());
                }
                return this;
            }
        });
        folderFileList.addListSelectionListener(e -> {
            if (e.getValueIsAdjusting()) return;
            File selected = folderFileList.getSelectedValue();
            if (selected == null) return;
            if (loadImageFile(selected) && statusLabel != null) {
                int index = folderFileList.getSelectedIndex() + 1;
                int total = folderFileListModel.getSize();
                String folderName = currentFolder != null ? currentFolder.getName() : selected.getParentFile().getName();
                statusLabel.setText(" Folder: " + folderName + " | " + index + "/" + total + " | " + selected.getName());
            }
        });

        folderListScrollPane = new JScrollPane(folderFileList);
        folderListScrollPane.setBorder(BorderFactory.createTitledBorder("Folder Files"));
        folderListScrollPane.setPreferredSize(new Dimension(240, 0));
        folderListScrollPane.setMinimumSize(new Dimension(180, 0));

        emptyFolderPanel = new JPanel();
        fileBrowserSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, emptyFolderPanel, imageScrollPane);
        fileBrowserSplitPane.setResizeWeight(0.0);
        fileBrowserSplitPane.setContinuousLayout(true);
        fileBrowserSplitPane.setOneTouchExpandable(true);
        panel.add(fileBrowserSplitPane, BorderLayout.CENTER);
        setFolderListVisible(false);

        JPanel zoomPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        JButton zoomInBtn = new JButton("+");
        JButton zoomOutBtn = new JButton("-");
        JButton zoomFitBtn = new JButton("Fit");
        JLabel zoomLabel = new JLabel("100%");

        zoomInBtn.addActionListener(e -> { zoomIn(); zoomLabel.setText(String.format("%.0f%%", zoomFactor * 100)); });
        zoomOutBtn.addActionListener(e -> { zoomOut(); zoomLabel.setText(String.format("%.0f%%", zoomFactor * 100)); });
        zoomFitBtn.addActionListener(e -> { zoomFit(); zoomLabel.setText("Fit"); });

        zoomPanel.add(new JLabel("Zoom:"));
        zoomPanel.add(zoomOutBtn);
        zoomPanel.add(zoomLabel);
        zoomPanel.add(zoomInBtn);
        zoomPanel.add(zoomFitBtn);

        panel.add(zoomPanel, BorderLayout.SOUTH);

        return panel;
    }

    private JPanel createSection(String title) {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), title,
            TitledBorder.LEFT, TitledBorder.TOP,
            new Font("Arial", Font.BOLD, 12)));
        panel.setAlignmentX(Component.LEFT_ALIGNMENT);
        return panel;
    }

    private void addLabeledComponent(JPanel panel, String label, JComponent component) {
        JPanel row = new JPanel(new BorderLayout(5, 0));
        row.setMaximumSize(new Dimension(Integer.MAX_VALUE, 30));
        row.setAlignmentX(Component.LEFT_ALIGNMENT);
        JLabel jLabel = new JLabel(label);
        jLabel.setPreferredSize(new Dimension(150, 25));
        row.add(jLabel, BorderLayout.WEST);
        row.add(component, BorderLayout.CENTER);
        panel.add(row);
        panel.add(Box.createRigidArea(new Dimension(0, 5)));
    }

    private void initializeColorMap() {
        colorMap = new Color[256];
        for (int i = 0; i < 256; i++) {
            float hue = (i * 137.508f) % 360 / 360.0f;
            colorMap[i] = Color.getHSBColor(hue, 0.8f, 0.9f);
        }
        colorMap[0] = new Color(0, 0, 0, 0);
    }

    private void loadModels() {
        SwingWorker<JSONObject, Void> worker = new SwingWorker<>() {
            @Override
            protected JSONObject doInBackground() throws Exception {
                JSONObject models = new JSONObject();

                java.util.LinkedHashSet<String> cellpose31Set = new java.util.LinkedHashSet<>();
                java.util.LinkedHashSet<String> cellposeSAMSet = new java.util.LinkedHashSet<>();

                cellpose31Set.add("cyto3");

                if (backendRunner.modelsDir != null) {
                    Path cellpose31Dir = backendRunner.modelsDir.resolve("Cellpose 3.1");
                    if (cellpose31Dir.toFile().exists()) {
                        File[] files = cellpose31Dir.toFile().listFiles();
                        if (files != null) {
                            for (File file : files) {
                                if (file.isFile() && !file.getName().equals("README.md")) {
                                    cellpose31Set.add(file.getName());
                                }
                            }
                        }
                    }
                }

                JSONArray cellpose31Models = new JSONArray();
                for (String model : cellpose31Set) {
                    cellpose31Models.put(model);
                }
                models.put("Cellpose3.1", cellpose31Models);

                cellposeSAMSet.add("cpsam");

                if (backendRunner.modelsDir != null) {
                    Path cellposeSAMDir = backendRunner.modelsDir.resolve("CellposeSAM");
                    if (cellposeSAMDir.toFile().exists()) {
                        File[] files = cellposeSAMDir.toFile().listFiles();
                        if (files != null) {
                            for (File file : files) {
                                if (file.isFile() && !file.getName().equals("README.md")) {
                                    cellposeSAMSet.add(file.getName());
                                }
                            }
                        }
                    }
                }

                JSONArray cellposeSAMModels = new JSONArray();
                for (String model : cellposeSAMSet) {
                    cellposeSAMModels.put(model);
                }
                models.put("CellposeSAM", cellposeSAMModels);

                return models;
            }

            @Override
            protected void done() {
                try {
                    JSONObject models = get();
                    String modelType = (String) modelTypeCombo.getSelectedItem();
                    JSONArray modelList = models.getJSONArray(modelType);

                    modelNameCombo.removeAllItems();
                    for (int i = 0; i < modelList.length(); i++) {
                        modelNameCombo.addItem(modelList.getString(i));
                    }

                    int modelCount = modelList.length();
                    statusLabel.setText(modelCount > 0
                        ? " " + modelCount + " model(s) available"
                        : " No models found");
                } catch (Exception e) {
                    statusLabel.setText(" Error loading models: " + e.getMessage());
                    modelNameCombo.removeAllItems();
                    modelNameCombo.addItem("cyto3");
                }
            }
        };
        worker.execute();
    }

    private void openImage() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter(
            "Image Files", "tif", "tiff", "png", "jpg", "jpeg", "bmp"));

        if (fileChooser.showOpenDialog(frame) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            if (loadImageFile(file)) {
                currentFolder = null;
                folderFileListModel.clear();
                setFolderListVisible(false);
                segmentAllButton.setEnabled(false);
                classifyAllButton.setEnabled(false);
                statusLabel.setText(" Loaded: " + file.getName());
            }
        }
    }

    private void openFolder() {
        JFileChooser folderChooser = new JFileChooser();
        folderChooser.setDialogTitle("Select Folder with Images");
        folderChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        folderChooser.setAcceptAllFileFilterUsed(false);

        if (folderChooser.showOpenDialog(frame) != JFileChooser.APPROVE_OPTION) {
            return;
        }

        File folder = folderChooser.getSelectedFile();
        File[] candidates = folder.listFiles((dir, name) -> isSupportedImageName(name));

        if (candidates == null || candidates.length == 0) {
            JOptionPane.showMessageDialog(frame,
                "No supported images found in folder:\n" + folder.getAbsolutePath(),
                "Open Folder", JOptionPane.WARNING_MESSAGE);
            return;
        }

        Arrays.sort(candidates, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
        currentFolder = folder;
        folderFileListModel.clear();
        for (File candidate : candidates) {
            folderFileListModel.addElement(candidate);
        }
        setFolderListVisible(true);
        segmentAllButton.setEnabled(true);
        classifyAllButton.setEnabled(true);
        folderFileList.setSelectedIndex(0);
        folderFileList.ensureIndexIsVisible(0);
    }

    private boolean isSupportedImageName(String name) {
        String lower = name.toLowerCase();
        boolean supportedExtension = lower.endsWith(".tif")
            || lower.endsWith(".tiff")
            || lower.endsWith(".png")
            || lower.endsWith(".jpg")
            || lower.endsWith(".jpeg")
            || lower.endsWith(".bmp");
        return supportedExtension && !isGeneratedMaskName(lower);
    }

    private boolean isGeneratedMaskName(String lowerName) {
        if (lowerName.endsWith("_cp_masks.png")) {
            return true;
        }
        int dot = lowerName.lastIndexOf('.');
        if (dot <= 0) return false;
        return lowerName.substring(0, dot).endsWith("_mask");
    }

    private void setFolderListVisible(boolean visible) {
        if (fileBrowserSplitPane == null) return;
        if (visible) {
            fileBrowserSplitPane.setLeftComponent(folderListScrollPane);
            fileBrowserSplitPane.setDividerSize(8);
            if (fileBrowserSplitPane.getDividerLocation() < 120) {
                fileBrowserSplitPane.setDividerLocation(240);
            }
        } else {
            fileBrowserSplitPane.setLeftComponent(emptyFolderPanel);
            fileBrowserSplitPane.setDividerSize(0);
            fileBrowserSplitPane.setDividerLocation(0);
        }
        fileBrowserSplitPane.revalidate();
        fileBrowserSplitPane.repaint();
    }

    private boolean loadImageFile(File file) {
        try {
            Opener opener = new Opener();
            imagePlus = opener.openImage(file.getAbsolutePath());
            if (imagePlus != null) {
                originalImage = imagePlus.getBufferedImage();
            } else {
                originalImage = ImageIO.read(file);
            }

            if (originalImage != null) {
                currentImageFile = file;
                classificationOverlay = null;
                fociMaskImage = null;
                classificationResults.clear();
                cellCentroids.clear();
                if (classificationSummaryLabel != null) {
                    classificationSummaryLabel.setText("No classification results");
                }

                maskImage = null;
                File existingMask = OutputPaths.buildMaskOutputFile(file);
                if (existingMask.exists()) {
                    try {
                        BufferedImage rawMask = ImageIO.read(existingMask);
                        if (rawMask != null) {
                            int[] numCellsOut = {0};
                            maskImage = MaskUtils.createColoredMaskFromLabels(rawMask, numCellsOut, colorMap);
                            cellCentroids = MaskUtils.computeCellCentroidsFromMask(existingMask);
                        }
                    } catch (Exception ex) {
                        System.err.println("[Cellpose] Could not reload mask: " + ex.getMessage());
                    }
                }

                reloadClassificationResults(file, existingMask);
                zoomFit();
                return true;
            }

            JOptionPane.showMessageDialog(frame,
                "Failed to load image: " + file.getName(),
                "Load Error", JOptionPane.ERROR_MESSAGE);
            return false;
        } catch (Exception e) {
            JOptionPane.showMessageDialog(frame,
                "Error loading image: " + e.getMessage(),
                "Load Error", JOptionPane.ERROR_MESSAGE);
            return false;
        }
    }

    private void runSegmentation() {
        if (currentImageFile == null || originalImage == null) {
            JOptionPane.showMessageDialog(frame,
                "Please load an image first.", "No Image", JOptionPane.WARNING_MESSAGE);
            return;
        }
        List<File> singleFile = new ArrayList<>();
        singleFile.add(currentImageFile);
        executeSegmentationBatch(singleFile, false);
    }

    private void runBatchSegmentation() {
        if (currentFolder == null || folderFileListModel == null || folderFileListModel.getSize() == 0) {
            JOptionPane.showMessageDialog(frame,
                "Please open a folder first.", "No Folder", JOptionPane.WARNING_MESSAGE);
            return;
        }

        List<File> files = getFilesForSegmentation();
        if (files.isEmpty()) {
            JOptionPane.showMessageDialog(frame,
                "No valid images found in folder.", "No Images", JOptionPane.WARNING_MESSAGE);
            return;
        }

        int confirm = JOptionPane.showConfirmDialog(frame,
            "Run segmentation for all " + files.size() + " images in this folder?",
            "Batch Segmentation", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
        if (confirm != JOptionPane.OK_OPTION) return;

        executeSegmentationBatch(files, true);
    }

    private void executeSegmentationBatch(List<File> taskFiles, boolean isBatch) {
        if (isBatch) {
            segmentAllButton.setEnabled(false);
        } else {
            segmentButton.setEnabled(false);
        }
        progressBar.setIndeterminate(true);
        progressBar.setString("Running segmentation...");
        statusLabel.setText(" Running segmentation...");
        segmentationSummaryLabel.setText("Segmenting...");

        SwingWorker<BufferedImage, String> worker = new SwingWorker<>() {
            private int savedMasksCount = 0;
            private int noMasksCount = 0;

            @Override
            protected BufferedImage doInBackground() throws Exception {
                String modelType = (String) modelTypeCombo.getSelectedItem();
                String modelName = (String) modelNameCombo.getSelectedItem();
                String diameter = String.valueOf(diameterSpinner.getValue());
                String channels = channelsField.getText();
                boolean useGpu = useGpuCheckbox.isSelected();
                String flowThreshold = String.valueOf(flowThresholdSpinner.getValue());
                String cellprobThreshold = String.valueOf(cellprobThresholdSpinner.getValue());

                if (modelType == null || modelName == null) {
                    throw new Exception("Please select a valid model type and model.");
                }

                String pretrainedModelArg = backendRunner.resolvePretrainedModelArg(modelType, modelName);
                int[] parsedChannels = BackendRunner.parseChannels(channels);

                publish("Initializing " + modelType + "/" + modelName + "...");

                List<ImageCellStats> allImageStats = new ArrayList<>();
                BufferedImage previewMask = null;
                int total = taskFiles.size();
                for (int i = 0; i < total; i++) {
                    File imageFile = taskFiles.get(i);
                    publish("[" + (i + 1) + "/" + total + "] Segmenting " + imageFile.getName() + "...");

                    SegmentationResult result = backendRunner.segmentImageFile(
                        imageFile, modelType, pretrainedModelArg,
                        diameter, flowThreshold, cellprobThreshold, useGpu, parsedChannels
                    );

                    if (result.noMasksFound) {
                        noMasksCount++;
                        publish("No masks found for " + imageFile.getName() + "; skipping.");
                        continue;
                    }

                    int[] numCellsOut = {0};
                    BufferedImage coloredMask = MaskUtils.createColoredMaskFromLabels(result.rawMask, numCellsOut, colorMap);

                    File outputMaskFile = OutputPaths.buildMaskOutputFile(imageFile);
                    ImageIO.write(result.rawMask, "png", outputMaskFile);
                    File outputPropertiesFile = OutputPaths.buildMaskPropertiesOutputFile(imageFile);
                    ImageCellStats imgStats = writeMaskPropertiesCsv(result.rawMask, outputPropertiesFile, imageFile.getName());
                    allImageStats.add(imgStats);
                    savedMasksCount++;
                    publish("Saved " + outputMaskFile.getName() + " and "
                        + outputPropertiesFile.getName() + " (" + numCellsOut[0] + " cells)");

                    if (currentImageFile != null && currentImageFile.equals(imageFile)) {
                        previewMask = coloredMask;
                    } else if (previewMask == null) {
                        previewMask = coloredMask;
                    }
                }

                if (!allImageStats.isEmpty() && isBatch) {
                    File outputsRoot = OutputPaths.getOutputsRootDir(taskFiles.get(0));
                    writeOverallPropertiesCsv(allImageStats, new File(outputsRoot, "overall_properties.csv"));
                    publish("Saved overall_properties.csv");
                }

                return previewMask;
            }

            @Override
            protected void process(java.util.List<String> chunks) {
                for (String msg : chunks) {
                    statusLabel.setText(" " + msg);
                }
            }

            @Override
            protected void done() {
                try {
                    maskImage = get();
                    classificationOverlay = null;
                    classificationResults.clear();
                    cellCentroids.clear();
                    File maskFile = currentImageFile != null ? OutputPaths.buildMaskOutputFile(currentImageFile) : null;
                    if (maskFile != null && maskFile.exists()) {
                        cellCentroids = MaskUtils.computeCellCentroidsFromMask(maskFile);
                    }
                    updateDisplay();

                    String summaryText;
                    if (isBatch) {
                        String folderName = currentFolder != null ? currentFolder.getName() : "folder";
                        summaryText = savedMasksCount + " segmented, " + noMasksCount + " no masks ("
                            + taskFiles.size() + " images in " + folderName + ")";
                    } else if (noMasksCount > 0 && currentImageFile != null) {
                        summaryText = "No masks found for " + currentImageFile.getName();
                    } else if (currentImageFile != null) {
                        summaryText = currentImageFile.getName() + ": " + cellCentroids.size() + " cells segmented";
                    } else {
                        summaryText = "Segmentation completed";
                    }
                    segmentationSummaryLabel.setText(summaryText);
                    statusLabel.setText(" " + summaryText);
                    progressBar.setString("Completed");
                } catch (Exception e) {
                    statusLabel.setText(" Segmentation failed: " + e.getMessage());
                    segmentationSummaryLabel.setText("Segmentation failed");
                    progressBar.setString("Failed");
                    JOptionPane.showMessageDialog(frame,
                        "Segmentation error: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
                }
                progressBar.setIndeterminate(false);
                if (isBatch) {
                    segmentAllButton.setEnabled(true);
                } else {
                    segmentButton.setEnabled(true);
                }
            }
        };
        worker.execute();
    }

    private List<File> getFilesForSegmentation() {
        List<File> files = new ArrayList<>();

        if (currentFolder != null && folderFileListModel != null && folderFileListModel.getSize() > 0) {
            for (int i = 0; i < folderFileListModel.getSize(); i++) {
                File file = folderFileListModel.getElementAt(i);
                if (file != null && file.isFile() && isSupportedImageName(file.getName())) {
                    files.add(file);
                }
            }
            return files;
        }

        if (currentImageFile != null && currentImageFile.isFile()) {
            files.add(currentImageFile);
        }
        return files;
    }

    private ImageCellStats writeMaskPropertiesCsv(BufferedImage rawMask, File csvFile, String imageName) throws IOException {
        int width = rawMask.getWidth();
        int height = rawMask.getHeight();
        Raster raster = rawMask.getRaster();

        Map<Integer, MaskPropertiesAccumulator> statsByLabel = new TreeMap<>();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int label = MaskUtils.getMaskLabel(raster, x, y);
                if (label <= 0) continue;
                statsByLabel.computeIfAbsent(label, MaskPropertiesAccumulator::new).addPixel(x, y);
            }
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int label = MaskUtils.getMaskLabel(raster, x, y);
                if (label <= 0) continue;
                MaskPropertiesAccumulator stats = statsByLabel.get(label);
                if (stats != null) {
                    stats.sumXX += (double) x * x;
                    stats.sumYY += (double) y * y;
                    stats.sumXY += (double) x * y;
                    if (MaskUtils.isBoundaryPixel(raster, width, height, x, y, label)) {
                        stats.boundaryPixels++;
                    }
                }
            }
        }

        try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(
            new FileOutputStream(csvFile), StandardCharsets.UTF_8
        ))) {
            writer.println("label,area_pixels,centroid_x,centroid_y,bbox_min_x,bbox_min_y,bbox_max_x,bbox_max_y,boundary_pixels,equivalent_diameter,eccentricity");
            for (MaskPropertiesAccumulator stats : statsByLabel.values()) {
                writer.println(String.format(Locale.US,
                    "%d,%d,%.4f,%.4f,%d,%d,%d,%d,%d,%.4f,%.4f",
                    stats.label, stats.area,
                    stats.centroidX(), stats.centroidY(),
                    stats.minX, stats.minY, stats.maxX, stats.maxY,
                    stats.boundaryPixels,
                    stats.equivalentDiameter(), stats.eccentricity()));
            }
        }

        if (statsByLabel.isEmpty()) {
            return new ImageCellStats(imageName, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        double totalArea = 0, totalDiam = 0, totalEcc = 0;
        double minA = Double.MAX_VALUE, maxA = 0;
        double minD = Double.MAX_VALUE, maxD = 0;

        for (MaskPropertiesAccumulator stats : statsByLabel.values()) {
            double a = stats.area;
            double d = stats.equivalentDiameter();
            double e = stats.eccentricity();
            totalArea += a;
            totalDiam += d;
            totalEcc += e;
            if (a < minA) minA = a;
            if (a > maxA) maxA = a;
            if (d < minD) minD = d;
            if (d > maxD) maxD = d;
        }

        int n = statsByLabel.size();
        return new ImageCellStats(imageName, n,
            totalArea / n, minA, maxA,
            totalDiam / n, minD, maxD,
            totalEcc / n);
    }

    private void writeOverallPropertiesCsv(List<ImageCellStats> allStats, File csvFile) throws IOException {
        try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(
            new FileOutputStream(csvFile), StandardCharsets.UTF_8
        ))) {
            writer.println("image,cell_count,avg_area,min_area,max_area,min_diameter,max_diameter,avg_diameter,avg_eccentricity");

            double sumAvgArea = 0, sumAvgDiam = 0, sumAvgEcc = 0;
            double globalMinArea = Double.MAX_VALUE, globalMaxArea = 0;
            double globalMinDiam = Double.MAX_VALUE, globalMaxDiam = 0;
            int totalCells = 0;

            for (ImageCellStats s : allStats) {
                writer.println(String.format(Locale.US,
                    "%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                    s.imageName, s.cellCount,
                    s.avgArea, s.minArea, s.maxArea,
                    s.minDiameter, s.maxDiameter, s.avgDiameter,
                    s.avgEccentricity));

                totalCells += s.cellCount;
                sumAvgArea += s.avgArea;
                sumAvgDiam += s.avgDiameter;
                sumAvgEcc += s.avgEccentricity;
                if (s.minArea < globalMinArea) globalMinArea = s.minArea;
                if (s.maxArea > globalMaxArea) globalMaxArea = s.maxArea;
                if (s.minDiameter < globalMinDiam) globalMinDiam = s.minDiameter;
                if (s.maxDiameter > globalMaxDiam) globalMaxDiam = s.maxDiameter;
            }

            int n = allStats.size();
            writer.println(String.format(Locale.US,
                "OVERALL,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                totalCells,
                sumAvgArea / n, globalMinArea, globalMaxArea,
                globalMinDiam, globalMaxDiam, sumAvgDiam / n, sumAvgEcc / n));
        }
        System.out.println("[Cellpose] Saved overall properties: " + csvFile.getAbsolutePath());
    }

    private void updateChannelSelectionState() {
        if (channelsField == null || modelTypeCombo == null) return;
        String modelType = (String) modelTypeCombo.getSelectedItem();
        boolean enableChannels = "Cellpose3.1".equals(modelType);
        channelsField.setEnabled(enableChannels);
        if (enableChannels) {
            channelsField.setToolTipText("Cellpose3.1: 0=GRAY, 1=RED, 2=GREEN, 3=BLUE (format: chan,chan2)");
        } else {
            channelsField.setToolTipText("CellposeSAM ignores --chan/--chan2 in CLI (v4.0.1+)");
        }
    }

    private void updateDisplay() {
        if (originalImage == null) return;

        displayImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = displayImage.createGraphics();
        g.drawImage(originalImage, 0, 0, null);

        if (maskImage != null && showMask) {
            g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, maskOpacity));
            g.drawImage(maskImage, 0, 0, null);
        }

        if (classificationOverlay != null && showClassification) {
            g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, maskOpacity));
            g.drawImage(classificationOverlay, 0, 0, null);
        }

        if (fociMaskImage != null && showFociMask) {
            g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.7f));
            int fw = fociMaskImage.getWidth();
            int fh = fociMaskImage.getHeight();
            BufferedImage fociColored = new BufferedImage(fw, fh, BufferedImage.TYPE_INT_ARGB);
            for (int fy = 0; fy < fh; fy++) {
                for (int fx = 0; fx < fw; fx++) {
                    if ((fociMaskImage.getRGB(fx, fy) & 0xFF) > 0) {
                        fociColored.setRGB(fx, fy, new Color(0, 255, 255, 200).getRGB());
                    }
                }
            }
            g.drawImage(fociColored, 0, 0, null);
        }

        if (showLabels && !cellCentroids.isEmpty()) {
            g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
            int fontSize = Math.max(10, Math.min(16, originalImage.getWidth() / 80));
            g.setFont(new Font("Arial", Font.BOLD, fontSize));
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            FontMetrics fm = g.getFontMetrics();

            for (Map.Entry<Integer, int[]> entry : cellCentroids.entrySet()) {
                int label = entry.getKey();
                int[] centroid = entry.getValue();
                int cx = centroid[0];
                int cy = centroid[1];
                String text = String.valueOf(label);
                int tw = fm.stringWidth(text);
                int th = fm.getAscent();
                g.setColor(new Color(0, 0, 0, 160));
                g.fillRoundRect(cx - tw / 2 - 3, cy - th / 2 - 2, tw + 6, th + 4, 6, 6);
                g.setColor(Color.WHITE);
                g.drawString(text, cx - tw / 2, cy + th / 2 - 1);
            }
        }
        g.dispose();

        int width = (int) (displayImage.getWidth() * zoomFactor);
        int height = (int) (displayImage.getHeight() * zoomFactor);
        Image scaledImage = displayImage.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        imageLabel.setIcon(new ImageIcon(scaledImage));
        imageLabel.setText(null);
    }

    private void zoomIn() {
        zoomFactor *= 1.25;
        updateDisplay();
    }

    private void zoomOut() {
        zoomFactor /= 1.25;
        updateDisplay();
    }

    private void zoomFit() {
        if (originalImage != null) {
            Dimension viewportSize = imageLabel.getParent().getSize();
            double scaleX = (double) viewportSize.width / originalImage.getWidth();
            double scaleY = (double) viewportSize.height / originalImage.getHeight();
            zoomFactor = Math.min(scaleX, scaleY) * 0.95;
            updateDisplay();
        }
    }

    private void exportMask() {
        if (maskImage == null) {
            JOptionPane.showMessageDialog(frame, "No mask to export", "Export", JOptionPane.WARNING_MESSAGE);
            return;
        }
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("PNG Images", "png"));
        if (fileChooser.showSaveDialog(frame) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            if (!file.getName().toLowerCase().endsWith(".png")) {
                file = new File(file.getAbsolutePath() + ".png");
            }
            try {
                ImageIO.write(maskImage, "png", file);
                statusLabel.setText(" Mask exported: " + file.getName());
            } catch (Exception e) {
                JOptionPane.showMessageDialog(frame, "Export failed: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void exportOverlay() {
        if (displayImage == null) {
            JOptionPane.showMessageDialog(frame, "No image to export", "Export", JOptionPane.WARNING_MESSAGE);
            return;
        }
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("PNG Images", "png"));
        if (fileChooser.showSaveDialog(frame) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            if (!file.getName().toLowerCase().endsWith(".png")) {
                file = new File(file.getAbsolutePath() + ".png");
            }
            try {
                ImageIO.write(displayImage, "png", file);
                statusLabel.setText(" Overlay exported: " + file.getName());
            } catch (Exception e) {
                JOptionPane.showMessageDialog(frame, "Export failed: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────────────
    //  FOCI CLASSIFICATION
    // ───────────────────────────────────────────────────────────────────────────

    private void loadClassificationModels() {
        if (classificationModelCombo == null) return;
        classificationModelCombo.removeAllItems();

        if (backendRunner.modelsDir == null) {
            classificationModelCombo.addItem("(no models directory)");
            return;
        }

        Path dinoDir = backendRunner.modelsDir.resolve("DINO");
        if (!dinoDir.toFile().exists() || !dinoDir.toFile().isDirectory()) {
            classificationModelCombo.addItem("(no DINO models found)");
            return;
        }

        File[] modelFiles = dinoDir.toFile().listFiles((dir, name) -> name.endsWith(".pth"));
        if (modelFiles == null || modelFiles.length == 0) {
            classificationModelCombo.addItem("(no .pth files found)");
            return;
        }

        Arrays.sort(modelFiles, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
        for (File f : modelFiles) {
            classificationModelCombo.addItem(f.getName());
        }
    }

    private void runClassification() {
        if (currentImageFile == null || originalImage == null) {
            JOptionPane.showMessageDialog(frame,
                "Please load an image first.", "No Image", JOptionPane.WARNING_MESSAGE);
            return;
        }

        File maskFile = findMaskFileForCurrentImage();
        if (maskFile == null || !maskFile.exists()) {
            JOptionPane.showMessageDialog(frame,
                "No segmentation mask found.\nPlease run segmentation first.",
                "No Mask", JOptionPane.WARNING_MESSAGE);
            return;
        }

        classifyButton.setEnabled(false);
        classificationSummaryLabel.setText("Classifying...");
        executeSingleClassification(currentImageFile, maskFile, true);
    }

    private void runBatchClassification() {
        if (currentFolder == null || folderFileListModel == null || folderFileListModel.getSize() == 0) {
            JOptionPane.showMessageDialog(frame,
                "Please open a folder first.", "No Folder", JOptionPane.WARNING_MESSAGE);
            return;
        }

        List<File> filesToClassify = new ArrayList<>();
        List<File> missingMasks = new ArrayList<>();
        for (int i = 0; i < folderFileListModel.getSize(); i++) {
            File file = folderFileListModel.getElementAt(i);
            File mask = OutputPaths.buildMaskOutputFile(file);
            if (mask.exists()) {
                filesToClassify.add(file);
            } else {
                missingMasks.add(file);
            }
        }

        if (filesToClassify.isEmpty()) {
            JOptionPane.showMessageDialog(frame,
                "No segmentation masks found.\nPlease run 'Segment All' first.",
                "No Masks", JOptionPane.WARNING_MESSAGE);
            return;
        }

        String msg = "Classify " + filesToClassify.size() + " images?";
        if (!missingMasks.isEmpty()) {
            msg += "\n(" + missingMasks.size() + " images have no mask and will be skipped)";
        }
        int confirm = JOptionPane.showConfirmDialog(frame, msg,
            "Batch Classification", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
        if (confirm != JOptionPane.OK_OPTION) return;

        classifyAllButton.setEnabled(false);
        statusLabel.setText(" Running batch classification...");
        classificationSummaryLabel.setText("Classifying...");

        // Capture all params on the EDT before handing off to the worker thread
        String method = (String) classificationMethodCombo.getSelectedItem();
        int fociChannel = ((Number) fociChannelSpinner.getValue()).intValue();
        String selectedModel = (String) classificationModelCombo.getSelectedItem();
        Path modelPath = (backendRunner.modelsDir != null && selectedModel != null && !selectedModel.startsWith("("))
            ? backendRunner.modelsDir.resolve("DINO").resolve(selectedModel) : null;
        boolean useGpu = useGpuCheckbox.isSelected();
        int denoiseH = ((Number) denoiseHSpinner.getValue()).intValue();
        int tophatSize = ((Number) tophatSizeSpinner.getValue()).intValue();
        double clipLimit = ((Number) clipLimitSpinner.getValue()).doubleValue();
        int minArea = ((Number) minAreaSpinner.getValue()).intValue();
        int damageThreshold = ((Number) damageThresholdSpinner.getValue()).intValue();

        SwingWorker<Void, String> worker = new SwingWorker<>() {
            private int classified = 0;
            private int errors = 0;
            private int totalHealthy = 0;
            private int totalDamaged = 0;

            @Override
            protected Void doInBackground() throws Exception {
                int total = filesToClassify.size();
                for (int i = 0; i < total; i++) {
                    File imageFile = filesToClassify.get(i);
                    File maskFile = OutputPaths.buildMaskOutputFile(imageFile);
                    publish("[" + (i + 1) + "/" + total + "] Classifying " + imageFile.getName() + "...");

                    JSONObject result;
                    try {
                        if ("DINO".equals(method)) {
                            if (modelPath == null || !modelPath.toFile().exists()) {
                                throw new Exception("DINO model not found");
                            }
                            result = backendRunner.executeDinoClassification(imageFile, maskFile, modelPath.toFile(), fociChannel, useGpu);
                        } else {
                            result = backendRunner.executeThresholdClassification(imageFile, maskFile, fociChannel,
                                denoiseH, tophatSize, clipLimit, minArea, damageThreshold);
                        }
                    } catch (Exception e) {
                        errors++;
                        publish("Error classifying " + imageFile.getName() + ": " + e.getMessage());
                        continue;
                    }

                    if (result.has("error")) {
                        errors++;
                        publish("Error: " + result.getString("error"));
                        continue;
                    }

                    saveClassificationCsv(imageFile, result);
                    JSONObject summary = result.getJSONObject("summary");
                    totalHealthy += summary.getInt("healthy");
                    totalDamaged += summary.getInt("damaged");
                    classified++;
                    publish("Classified " + imageFile.getName()
                        + " (" + summary.getInt("healthy") + " healthy, "
                        + summary.getInt("damaged") + " damaged)");
                }
                return null;
            }

            @Override
            protected void process(java.util.List<String> chunks) {
                for (String m : chunks) {
                    statusLabel.setText(" " + m);
                }
            }

            @Override
            protected void done() {
                try {
                    get();
                    String summaryText = String.format(
                        "Batch: %d classified | Healthy: %d | Damaged: %d | Errors: %d",
                        classified, totalHealthy, totalDamaged, errors);
                    classificationSummaryLabel.setText(summaryText);
                    statusLabel.setText(" " + summaryText);

                    if (currentImageFile != null) {
                        File maskFile = OutputPaths.buildMaskOutputFile(currentImageFile);
                        reloadClassificationResults(currentImageFile, maskFile);
                        updateDisplay();
                    }
                } catch (Exception e) {
                    statusLabel.setText(" Batch classification failed: " + e.getMessage());
                    classificationSummaryLabel.setText("Error");
                    e.printStackTrace();
                }
                classifyAllButton.setEnabled(true);
            }
        };
        worker.execute();
    }

    private void executeSingleClassification(File imageFile, File maskFile, boolean updateUi) {
        String method = (String) classificationMethodCombo.getSelectedItem();
        int fociChannel = ((Number) fociChannelSpinner.getValue()).intValue();

        if ("DINO".equals(method)) {
            String selectedModel = (String) classificationModelCombo.getSelectedItem();
            if (selectedModel == null || selectedModel.startsWith("(")) {
                JOptionPane.showMessageDialog(frame,
                    "No classification model selected.\n"
                    + "Please place a .pth model file in:\n"
                    + (backendRunner.modelsDir != null ? backendRunner.modelsDir.resolve("DINO").toString() : "models/DINO/"),
                    "No Model", JOptionPane.WARNING_MESSAGE);
                classifyButton.setEnabled(true);
                return;
            }
            Path modelPath = backendRunner.modelsDir.resolve("DINO").resolve(selectedModel);
            if (!modelPath.toFile().exists()) {
                JOptionPane.showMessageDialog(frame,
                    "Model file not found: " + modelPath, "Missing Model", JOptionPane.ERROR_MESSAGE);
                classifyButton.setEnabled(true);
                return;
            }

            boolean useGpu = useGpuCheckbox.isSelected();
            statusLabel.setText(" Running DINO classification (channel=" + fociChannel + ")...");
            SwingWorker<JSONObject, String> worker = new SwingWorker<>() {
                @Override
                protected JSONObject doInBackground() throws Exception {
                    return backendRunner.executeDinoClassification(imageFile, maskFile, modelPath.toFile(), fociChannel, useGpu);
                }

                @Override
                protected void done() {
                    try {
                        processClassificationResult(get(), imageFile, maskFile, updateUi);
                    } catch (Exception e) {
                        statusLabel.setText(" Classification failed: " + e.getMessage());
                        classificationSummaryLabel.setText("Error");
                        e.printStackTrace();
                        classifyButton.setEnabled(true);
                    }
                }
            };
            worker.execute();
        } else {
            int denoiseH = ((Number) denoiseHSpinner.getValue()).intValue();
            int tophatSize = ((Number) tophatSizeSpinner.getValue()).intValue();
            double clipLimit = ((Number) clipLimitSpinner.getValue()).doubleValue();
            int minArea = ((Number) minAreaSpinner.getValue()).intValue();
            int damageThreshold = ((Number) damageThresholdSpinner.getValue()).intValue();

            statusLabel.setText(" Running threshold classification (channel=" + fociChannel + ")...");
            SwingWorker<JSONObject, String> worker = new SwingWorker<>() {
                @Override
                protected JSONObject doInBackground() throws Exception {
                    return backendRunner.executeThresholdClassification(imageFile, maskFile, fociChannel,
                        denoiseH, tophatSize, clipLimit, minArea, damageThreshold);
                }

                @Override
                protected void done() {
                    try {
                        processClassificationResult(get(), imageFile, maskFile, updateUi);
                    } catch (Exception e) {
                        statusLabel.setText(" Classification failed: " + e.getMessage());
                        classificationSummaryLabel.setText("Error");
                        e.printStackTrace();
                        classifyButton.setEnabled(true);
                    }
                }
            };
            worker.execute();
        }
    }

    private void processClassificationResult(JSONObject result, File imageFile, File maskFile, boolean updateUi) {
        if (result.has("error")) {
            String err = result.getString("error");
            statusLabel.setText(" Classification failed: " + err);
            classificationSummaryLabel.setText("Error: " + err);
            JOptionPane.showMessageDialog(frame,
                "Classification error:\n" + err, "Error", JOptionPane.ERROR_MESSAGE);
        } else {
            JSONObject summary = result.getJSONObject("summary");
            int total = summary.getInt("total");
            int healthy = summary.getInt("healthy");
            int damaged = summary.getInt("damaged");

            classificationResults.clear();
            JSONArray cells = result.getJSONArray("cells");
            for (int i = 0; i < cells.length(); i++) {
                JSONObject cell = cells.getJSONObject(i);
                classificationResults.put(cell.getInt("label"), cell);
            }

            buildClassificationOverlay(maskFile);

            fociMaskImage = null;
            if (result.has("foci_mask_path")) {
                try {
                    File fociMaskFile = new File(result.getString("foci_mask_path"));
                    if (fociMaskFile.exists()) {
                        fociMaskImage = ImageIO.read(fociMaskFile);
                    }
                } catch (Exception e) {
                    System.err.println("[Classify] Failed to load foci mask: " + e.getMessage());
                }
            }

            String summaryText = String.format("Total: %d | Healthy: %d | Damaged: %d", total, healthy, damaged);
            classificationSummaryLabel.setText(summaryText);
            statusLabel.setText(" Classification complete: " + summaryText);

            saveClassificationCsv(imageFile, result);
            if (updateUi) updateDisplay();
        }
        classifyButton.setEnabled(true);
    }

    private void reloadClassificationResults(File imageFile, File maskFile) {
        classificationOverlay = null;
        fociMaskImage = null;
        classificationResults.clear();

        if (imageFile == null) return;

        File csvFile = OutputPaths.buildClassificationCsvFile(imageFile);
        if (!csvFile.exists()) {
            if (classificationSummaryLabel != null) {
                classificationSummaryLabel.setText("No classification results");
            }
            return;
        }

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(csvFile), StandardCharsets.UTF_8))) {
            String header = reader.readLine();
            if (header == null) return;

            int healthy = 0, damaged = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 7) continue;
                JSONObject cell = new JSONObject();
                cell.put("label", Integer.parseInt(parts[0]));
                cell.put("prediction", parts[1]);
                cell.put("probability", Double.parseDouble(parts[2]));
                JSONObject bbox = new JSONObject();
                bbox.put("min_x", Integer.parseInt(parts[3]));
                bbox.put("min_y", Integer.parseInt(parts[4]));
                bbox.put("max_x", Integer.parseInt(parts[5]));
                bbox.put("max_y", Integer.parseInt(parts[6]));
                cell.put("bbox", bbox);
                classificationResults.put(cell.getInt("label"), cell);
                if ("Damaged".equals(parts[1])) damaged++; else healthy++;
            }

            if (!classificationResults.isEmpty() && maskFile != null && maskFile.exists()) {
                buildClassificationOverlay(maskFile);
                if (classificationSummaryLabel != null) {
                    classificationSummaryLabel.setText(String.format(
                        "Total: %d | Healthy: %d | Damaged: %d",
                        classificationResults.size(), healthy, damaged));
                }
            }
        } catch (Exception e) {
            System.err.println("[Classify] Failed to reload classification: " + e.getMessage());
        }
    }

    private File findMaskFileForCurrentImage() {
        if (currentImageFile == null) return null;
        File maskFile = OutputPaths.buildMaskOutputFile(currentImageFile);
        return maskFile.exists() ? maskFile : null;
    }

    private void buildClassificationOverlay(File maskFile) {
        try {
            BufferedImage maskImg = ImageIO.read(maskFile);
            if (maskImg == null) return;

            int width = maskImg.getWidth();
            int height = maskImg.getHeight();
            classificationOverlay = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            Raster raster = maskImg.getRaster();

            Color healthyColor = new Color(0, 200, 0, 180);
            Color damagedColor = new Color(220, 30, 30, 180);
            Color unknownColor = new Color(128, 128, 128, 100);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int label = MaskUtils.getMaskLabel(raster, x, y);
                    if (label <= 0) {
                        classificationOverlay.setRGB(x, y, 0);
                        continue;
                    }
                    JSONObject cellResult = classificationResults.get(label);
                    Color color;
                    if (cellResult != null) {
                        color = "Damaged".equals(cellResult.getString("prediction")) ? damagedColor : healthyColor;
                    } else {
                        color = unknownColor;
                    }
                    classificationOverlay.setRGB(x, y, color.getRGB());
                }
            }
        } catch (Exception e) {
            System.err.println("[Classify] Failed to build overlay: " + e.getMessage());
            classificationOverlay = null;
        }
    }

    private void saveClassificationCsv(File imageFile, JSONObject result) {
        try {
            File csvFile = OutputPaths.buildClassificationCsvFile(imageFile);
            try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(
                    new FileOutputStream(csvFile), StandardCharsets.UTF_8))) {
                writer.println("label,prediction,probability,bbox_min_x,bbox_min_y,bbox_max_x,bbox_max_y");
                JSONArray cells = result.getJSONArray("cells");
                for (int i = 0; i < cells.length(); i++) {
                    JSONObject cell = cells.getJSONObject(i);
                    JSONObject bbox = cell.getJSONObject("bbox");
                    writer.println(String.format(Locale.US, "%d,%s,%.4f,%d,%d,%d,%d",
                        cell.getInt("label"),
                        cell.getString("prediction"),
                        cell.getDouble("probability"),
                        bbox.getInt("min_x"), bbox.getInt("min_y"),
                        bbox.getInt("max_x"), bbox.getInt("max_y")));
                }
            }
            System.out.println("[Classify] Saved results to: " + csvFile.getAbsolutePath());
        } catch (Exception e) {
            System.err.println("[Classify] Failed to save CSV: " + e.getMessage());
        }
    }
}
