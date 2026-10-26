package ru.nlp.laba3.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.csv.CsvMapper;

/** Единая конфигурация JSON и CSV; после создания мапперы не перенастраиваются. */
@Configuration(proxyBeanMethods = false)
public class JsonConfiguration {

    /** Возвращает основной Jackson 2 mapper для CLI и файлов моделей. */
    @Bean
    @Primary
    public ObjectMapper jsonMapper() {
        return new ObjectMapper();
    }

    /** Возвращает отдельный mapper для CSV. */
    @Bean
    public CsvMapper csvMapper() {
        return new CsvMapper();
    }
}
