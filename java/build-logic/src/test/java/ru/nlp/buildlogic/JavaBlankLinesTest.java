package ru.nlp.buildlogic;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class JavaBlankLinesTest {

    @Test
    void preservesCommentsStringsAndTextBlocksWhileSeparatingRealStatements() {
        String source = """
                class Probe {
                    String words = "if (true) return 3;";
                    String multiline = ""\"
                            return in text block;
                            if (not code)
                            ""\";
                    int choose(boolean ready) {
                        // return in comment;
                        if (ready) {
                            return 1;
                        }
                        return 2;
                    }
                }
                """;

        String formatted = JavaBlankLines.format(source);

        assertThat(formatted).contains("String words = \"if (true) return 3;\";")
                .contains("// return in comment;\n\n        if (ready)")
                .contains("{\n            return 1;")
                .contains("}\n\n        return 2;")
                .contains("return in text block;\n            if (not code)");
        assertThat(JavaBlankLines.format(formatted)).isEqualTo(formatted);
    }

    @Test
    void separatesCompleteNestedBlocksAndLeavesInlineElseIfIntact() {
        String source = """
                class Probe {
                    void work(boolean a, boolean b) {
                        int value = 1;
                        if (a) {
                            while (b) {
                                break;
                            }
                        } else if (b) {
                            value = 2;
                        }
                        System.out.println(value);
                    }
                }
                """;
        String formatted = JavaBlankLines.format(source);

        assertThat(formatted).contains("int value = 1;\n\n        if (a)")
                .contains("} else if (b) {")
                .contains("}\n        System.out.println(value);");
        assertThat(JavaBlankLines.format(formatted)).isEqualTo(formatted);
    }

    @Test
    void compactsSingleStatementBodiesAndAdjacentClosingBraces() {
        String source = """
                class Probe {
                    int value() {

                        return 1;
                    }
                    void work(boolean ready) {
                        if (ready) {
                            while (ready) {
                                break;
                            }

                        }

                    }
                }
                """;
        String formatted = JavaBlankLines.format(source);
        assertThat(formatted).contains("int value() {\n        return 1;")
                .contains("break;\n            }\n        }\n    }");
        assertThat(JavaBlankLines.format(formatted)).isEqualTo(formatted);
    }
}
