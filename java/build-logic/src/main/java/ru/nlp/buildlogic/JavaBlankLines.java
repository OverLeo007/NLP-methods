package ru.nlp.buildlogic;

import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.tools.DiagnosticCollector;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.ToolProvider;

import com.sun.source.tree.BlockTree;
import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.Tree;
import com.sun.source.util.JavacTask;
import com.sun.source.util.TreeScanner;
import com.sun.source.util.Trees;

/** Дополнительные обязательные интервалы по AST; строки и комментарии не переписываются. */
public final class JavaBlankLines {

    private JavaBlankLines() {
    }

    /** Разделяет этапы, сохраняя компактность однострочных тел и соседних закрывающих скобок. */
    public static String format(String source) {
        var compiler = ToolProvider.getSystemJavaCompiler();
        var diagnostics = new DiagnosticCollector<JavaFileObject>();
        var file = new SimpleJavaFileObject(URI.create("string:///FormattingInput.java"), JavaFileObject.Kind.SOURCE) {

            @Override
            public CharSequence getCharContent(boolean ignoreEncodingErrors) {
                return source;
            }
        };
        var insertions = new TreeSet<Integer>();
        var removals = new TreeMap<Integer, Integer>();

        try (var manager = compiler.getStandardFileManager(diagnostics, null, null)) {
            var task = (JavacTask) compiler.getTask(
                    null,
                    manager,
                    diagnostics,
                    List.of("-proc:none", "--release", "25"),
                    null,
                    List.of(file)
            );
            var positions = Trees.instance(task).getSourcePositions();

            for (CompilationUnitTree unit : task.parse()) {
                new TreeScanner<Void, Void>() {

                    @Override
                    public Void visitBlock(BlockTree block, Void unused) {
                        int begin = (int) positions.getStartPosition(unit, block);
                        int end = (int) positions.getEndPosition(unit, block);

                        if (begin >= 0 && end > begin) {
                            if (block.getStatements().size() <= 1) {
                                compactOpening(source, begin, removals);
                            }
                            compactClosing(source, end - 1, removals);
                        }

                        return super.visitBlock(block, unused);
                    }

                    @Override
                    public Void scan(Tree tree, Void unused) {

                        if (tree != null && isControlStatement(tree)) {
                            addBefore(source, (int) positions.getStartPosition(unit, tree), insertions);

                            if (tree.getKind() != Tree.Kind.RETURN && tree.getKind() != Tree.Kind.IF) {
                                addAfter(source, (int) positions.getEndPosition(unit, tree), insertions);
                            }
                        }

                        return super.scan(tree, unused);
                    }
                }.scan(unit, null);
            }
        } catch (IOException exception) {
            throw new IllegalStateException("Не удалось проверить интервалы Java", exception);
        }

        var result = new StringBuilder(source);

        var offsets = new TreeSet<>(insertions);
        offsets.addAll(removals.keySet());

        for (int offset : offsets.descendingSet()) {
            if (removals.containsKey(offset)) {
                result.delete(offset, removals.get(offset));
            } else if (
                removals.entrySet().stream().noneMatch(entry -> offset >= entry.getKey() && offset < entry.getValue())
            ) {
                result.insert(offset, '\n');
            }
        }

        return result.toString();
    }

    private static boolean isControlStatement(Tree tree) {
        return switch (tree.getKind()) {
            case IF, FOR_LOOP, ENHANCED_FOR_LOOP, WHILE_LOOP, DO_WHILE_LOOP, TRY, SWITCH, RETURN -> true;
            default -> false;
        };
    }

    private static void addBefore(String source, int offset, TreeSet<Integer> insertions) {

        if (offset < 0) {
            return;
        }

        int start = source.lastIndexOf('\n', offset - 1) + 1;

        if (start == 0 || !source.substring(start, offset).isBlank()) {
            return;
        }

        int previous = source.lastIndexOf('\n', start - 2) + 1;

        String previousLine = source.substring(previous, start - 1).strip();

        if (!previousLine.isEmpty() && !previousLine.endsWith("{")) {
            insertions.add(start);
        }
    }

    private static void addAfter(String source, int offset, TreeSet<Integer> insertions) {

        if (offset < 0) {
            return;
        }

        int end = source.indexOf('\n', offset);

        if (end < 0 || !source.substring(offset, end).isBlank()) {
            return;
        }

        int next = source.indexOf('\n', end + 1);
        next = next < 0 ? source.length() : next;

        String nextLine = source.substring(end + 1, next).strip();

        if (!nextLine.isEmpty() && !nextLine.startsWith("}")) {
            insertions.add(end + 1);
        }
    }

    private static void compactOpening(String source, int brace, TreeMap<Integer, Integer> removals) {
        int newline = source.indexOf('\n', brace);

        if (newline >= 0 && source.substring(brace + 1, newline).isBlank()) {
            removeEmptyLines(source, newline + 1, removals);
        }
    }

    private static void compactClosing(String source, int brace, TreeMap<Integer, Integer> removals) {
        int previous = brace - 1;

        while (previous >= 0 && Character.isWhitespace(source.charAt(previous))) {
            previous--;
        }

        if (previous >= 0 && source.charAt(previous) == '}') {
            int newline = source.indexOf('\n', previous);

            if (newline >= 0) {
                removeEmptyLines(source, newline + 1, removals);
            }
        }
    }

    private static void removeEmptyLines(String source, int start, TreeMap<Integer, Integer> removals) {
        int end = start;
        int next;

        while ((next = source.indexOf('\n', end)) >= 0 && source.substring(end, next).isBlank()) {
            end = next + 1;
        }

        if (end > start) {
            removals.put(start, end);
        }
    }
}
