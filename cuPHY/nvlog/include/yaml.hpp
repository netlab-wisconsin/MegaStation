/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(YAML_HPP_INCLUDED_)
#define YAML_HPP_INCLUDED_

#include "yaml.h"
#include <stdexcept>
#include <system_error>
#include <memory>

namespace yaml
{
////////////////////////////////////////////////////////////////////////
// yaml::node
class node {
public:
    //------------------------------------------------------------------
    // node()
    node(yaml_document_t* d, yaml_node_t* n) :
        document_(d), node_(n) {}
    //------------------------------------------------------------------
    // type()
    yaml_node_type_t type() const { return node_->type; }
    //------------------------------------------------------------------
    // type_string()
    const char* type_string() const
    {
        const char* s = "Unknown";
        switch(type())
        {
        case YAML_NO_NODE: s = "YAML_NO_NODE"; break;
        case YAML_SCALAR_NODE: s = "YAML_SCALAR_NODE"; break;
        case YAML_SEQUENCE_NODE: s = "YAML_SEQUENCE_NODE"; break;
        case YAML_MAPPING_NODE: s = "YAML_MAPPING_NODE"; break;
        }
        return s;
    }
    //------------------------------------------------------------------
    // operator[]()
    // Used for accessing mapping values from a key string. Throws an
    // exception if the key does not exist.
    node operator[](const char* key)
    {
        if(YAML_MAPPING_NODE != type())
        {
            throw std::runtime_error(std::string("YAML invalid type: '") +
                                     std::string(type_string()) +
                                     std::string(key) +
                                     std::string("' node accessed as 'mapping' node."));
        }
        for(yaml_node_pair_t* node_pair = node_->data.mapping.pairs.start;
            node_pair < node_->data.mapping.pairs.top;
            ++node_pair)
        {
            yaml_node_t* nkey = yaml_document_get_node(document_, node_pair->key);
            if((strlen(key) == nkey->data.scalar.length) &&
               (0 == strcmp(reinterpret_cast<char*>(nkey->data.scalar.value), key)))
            {
                yaml_node_t* nvalue = yaml_document_get_node(document_, node_pair->value);
                return node(document_, nvalue);
            }
        }
        throw std::runtime_error(std::string("YAML invalid key: ") +
                                 std::string(key));
    }

    node child(size_t index)
    {
        if(YAML_SEQUENCE_NODE != type())
        {
            throw std::runtime_error(std::string("YAML invalid type: '") +
                                     std::string(type_string()) +
                                     std::string("' node accessed as 'sequence' node."));
        }
        size_t sz = (node_->data.sequence.items.top - node_->data.sequence.items.start);
        if(index > sz)
        {
            throw std::runtime_error(std::string("YAML invalid index: node has only ") +
                                     std::to_string(sz) +
                                     std::string(" elements."));
        }
        return node(document_,
                    yaml_document_get_node(document_,
                                           *(node_->data.sequence.items.start + index)));
    }

    //------------------------------------------------------------------
    // operator[]()
    // Uses for accessing sequence values given an index. Throws an
    // exception if the index is greater than the number of elements in
    // the sequence.
    node operator[](size_t index)
    {
        if(YAML_SEQUENCE_NODE != type())
        {
            throw std::runtime_error(std::string("YAML invalid type: '") +
                                     std::string(type_string()) +
                                     std::string("' node accessed as 'sequence' node."));
        }
        size_t sz = (node_->data.sequence.items.top - node_->data.sequence.items.start);
        if(index > sz)
        {
            throw std::runtime_error(std::string("YAML invalid index: node has only ") +
                                     std::to_string(sz) +
                                     std::string(" elements."));
        }
        return node(document_,
                    yaml_document_get_node(document_,
                                           *(node_->data.sequence.items.start + index)));
    }
    //------------------------------------------------------------------
    // length()
    // Returns the length of a node
    // NO_NODE: 0
    // SCALAR: strlen(text)
    // SEQUENCE: number of elements
    // MAPPING: number of key-value pairs
    size_t length()
    {
        switch(type())
        {
        default:
        case YAML_NO_NODE: return 0;
        case YAML_SCALAR_NODE: return node_->data.scalar.length;
        case YAML_SEQUENCE_NODE: return (node_->data.sequence.items.top - node_->data.sequence.items.start);
        case YAML_MAPPING_NODE: return (node_->data.mapping.pairs.top - node_->data.mapping.pairs.start);
        }
    }
    //------------------------------------------------------------------
    // key()
    // Retrieve a key for a mapping node
    std::string key(size_t index)
    {
        if(YAML_MAPPING_NODE != type())
        {
            throw std::runtime_error(std::string("YAML invalid type: '") +
                                     std::string(type_string()) +
                                     std::string("' node accessed as 'mapping' node for key query."));
        }
        size_t len = length();
        if(index > len)
        {
            throw std::runtime_error(std::string("YAML invalid index: mapping node has only ") +
                                     std::to_string(len) +
                                     std::string(" key-value pairs."));
        }
        yaml_node_pair_t* node_pair = node_->data.mapping.pairs.start + index;
        yaml_node_t*      n         = yaml_document_get_node(document_, node_pair->key);
        return std::string(reinterpret_cast<char*>(n->data.scalar.value), n->data.scalar.length);
    }

    //------------------------------------------------------------------
    // operator float()
    operator float();
    //------------------------------------------------------------------
    // operator int()
    operator int();
    //------------------------------------------------------------------
    // operator unsigned int()
    operator unsigned int();
    //------------------------------------------------------------------
    // operator uint8_t()
    operator uint8_t();
    //------------------------------------------------------------------
    // operator unsigned int()
    operator uint16_t();
    //------------------------------------------------------------------
    // operator unsigned int()
    operator uint64_t();
    //------------------------------------------------------------------
    // operator std::string()
    operator std::string();
    //------------------------------------------------------------------
    // as<>()
    template <typename T>
    T as();

    //------------------------------------------------------------------
    // has_key()
    bool has_key(const char* k) const
    {
        if(YAML_MAPPING_NODE != type())
        {
            throw std::runtime_error(std::string("YAML invalid type: '") +
                                     std::string(type_string()) +
                                     std::string("' node accessed as 'mapping' node."));
        }
        for(yaml_node_pair_t* node_pair = node_->data.mapping.pairs.start;
            node_pair < node_->data.mapping.pairs.top;
            ++node_pair)
        {
            yaml_node_t* nkey = yaml_document_get_node(document_, node_pair->key);
            if((strlen(k) == nkey->data.scalar.length) &&
               (0 == strcmp(reinterpret_cast<char*>(nkey->data.scalar.value), k)))
            {
                return true;
            }
        }
        return false;
    }

private:
    yaml_document_t* document_;
    yaml_node_t*     node_;
};

////////////////////////////////////////////////////////////////////////
// node::as<float>()
template <>
inline float node::as<float>()
{
    if(YAML_SCALAR_NODE != type())
    {
        throw std::runtime_error(std::string("YAML invalid type: '") +
                                 std::string(type_string()) +
                                 std::string("' node accessed as 'scalar' node."));
    }
    float       f;
    std::string s(reinterpret_cast<char*>(node_->data.scalar.value),
                  node_->data.scalar.length);
    if(1 != sscanf(s.c_str(), "%f", &f))
    {
        throw std::runtime_error(std::string("YAML scalar error: '") +
                                 s +
                                 std::string("' could not be read as 'float'"));
    }
    return f;
}
//------------------------------------------------------------------
// operator float()
inline node::operator float()
{
    return as<float>();
}

////////////////////////////////////////////////////////////////////////
// node::as<int>()
template <>
inline int node::as<int>()
{
    if(YAML_SCALAR_NODE != type())
    {
        throw std::runtime_error(std::string("YAML invalid type: '") +
                                 std::string(type_string()) +
                                 std::string("' node accessed as 'scalar' node."));
    }
    int         i;
    std::string s(reinterpret_cast<char*>(node_->data.scalar.value),
                  node_->data.scalar.length);
    if(1 != sscanf(s.c_str(), "%i", &i))
    {
        throw std::runtime_error(std::string("YAML scalar error: '") +
                                 s +
                                 std::string("' could not be read as 'int'"));
    }
    return i;
}
// operator int()
inline node::operator int()
{
    return as<int>();
}

////////////////////////////////////////////////////////////////////////
// node::as<uint8_t>()
template <>
inline uint8_t node::as<uint8_t>()
{
    if(YAML_SCALAR_NODE != type())
    {
        throw std::runtime_error(std::string("YAML invalid type: '") +
                                 std::string(type_string()) +
                                 std::string("' node accessed as 'scalar' node."));
    }
    uint8_t     i;
    std::string s(reinterpret_cast<char*>(node_->data.scalar.value),
                  node_->data.scalar.length);
    if(1 != sscanf(s.c_str(), "%hhu", &i))
    {
        throw std::runtime_error(std::string("YAML scalar error: '") +
                                 s +
                                 std::string("' could not be read as 'uint8_t'"));
    }
    return i;
}

// operator uint8_t()
inline node::operator uint8_t()
{
    return as<uint8_t>();
}

////////////////////////////////////////////////////////////////////////
// node::as<uint16_t>()
template <>
inline uint16_t node::as<uint16_t>()
{
    if(YAML_SCALAR_NODE != type())
    {
        throw std::runtime_error(std::string("YAML invalid type: '") +
                                 std::string(type_string()) +
                                 std::string("' node accessed as 'scalar' node."));
    }
    uint16_t    i;
    std::string s(reinterpret_cast<char*>(node_->data.scalar.value),
                  node_->data.scalar.length);
    if(1 != sscanf(s.c_str(), "%hu", &i))
    {
        throw std::runtime_error(std::string("YAML scalar error: '") +
                                 s +
                                 std::string("' could not be read as 'uint16_t'"));
    }
    return i;
}
// operator uint16_t()
inline node::operator uint16_t()
{
    return as<uint16_t>();
}

////////////////////////////////////////////////////////////////////////
// node::as<uint64_t>()
template <>
inline uint64_t node::as<uint64_t>()
{
    if(YAML_SCALAR_NODE != type())
    {
        throw std::runtime_error(std::string("YAML invalid type: '") +
                                 std::string(type_string()) +
                                 std::string("' node accessed as 'scalar' node."));
    }
    uint64_t    i;
    std::string s(reinterpret_cast<char*>(node_->data.scalar.value),
                  node_->data.scalar.length);
    if(1 != sscanf(s.c_str(), "%lu", &i))
    {
        throw std::runtime_error(std::string("YAML scalar error: '") +
                                 s +
                                 std::string("' could not be read as 'uint64_t'"));
    }
    return i;
}
// operator uint64_t()
inline node::operator uint64_t()
{
    return as<uint64_t>();
}
////////////////////////////////////////////////////////////////////////
// node::as<unsigned int>()
template <>
inline unsigned int node::as<unsigned int>()
{
    if(YAML_SCALAR_NODE != type())
    {
        throw std::runtime_error(std::string("YAML invalid type: '") +
                                 std::string(type_string()) +
                                 std::string("' node accessed as 'scalar' node."));
    }
    unsigned int u;
    std::string  s(reinterpret_cast<char*>(node_->data.scalar.value),
                  node_->data.scalar.length);
    if(1 != sscanf(s.c_str(), "%u", &u))
    {
        throw std::runtime_error(std::string("YAML scalar error: '") +
                                 s +
                                 std::string("' could not be read as 'unsigned int'"));
    }
    return u;
}
// operator unsigned int()
inline node::operator unsigned int()
{
    return as<unsigned int>();
}

////////////////////////////////////////////////////////////////////////
// node::as<string>()
template <>
inline std::string node::as<std::string>()
{
    if(YAML_SCALAR_NODE != type())
    {
        throw std::runtime_error(std::string("YAML invalid type: '") +
                                 std::string(type_string()) +
                                 std::string("' node accessed as 'scalar' node."));
    }
    return std::string(reinterpret_cast<char*>(node_->data.scalar.value),
                       node_->data.scalar.length);
}
// operator std::string()
inline node::operator std::string()
{
    return as<std::string>();
}

////////////////////////////////////////////////////////////////////////
// yaml::document
class document {
public:
    //------------------------------------------------------------------
    // document()
    document(yaml_document_t* p = nullptr) :
        doc_ptr_(p, &document_destroy)
    {
    }
    //------------------------------------------------------------------
    // document()
    document(document&& d) :
        doc_ptr_(std::move(d.doc_ptr_))
    {
    }
    //------------------------------------------------------------------
    // ~document()
    ~document() = default;
    //------------------------------------------------------------------
    // document()
    document& operator=(document&& d)
    {
        doc_ptr_ = std::move(d.doc_ptr_);
        return *this;
    }
    //------------------------------------------------------------------
    // is_valid()
    bool is_valid() const { return (get_root_node() != nullptr); }
    //------------------------------------------------------------------
    // root()
    node root() { return node(doc_ptr_.get(), get_root_node()); }

private:
    yaml_node_t* get_root_node() const
    {
        return yaml_document_get_root_node(doc_ptr_.get());
    }
    //------------------------------------------------------------------
    static void document_destroy(yaml_document_t* d)
    {
        yaml_document_delete(d);
        delete d;
    }
    //------------------------------------------------------------------
    // Data
    std::unique_ptr<yaml_document_t, decltype(&document_destroy)> doc_ptr_;
};

////////////////////////////////////////////////////////////////////////
// yaml::parser
class parser {
public:
    //------------------------------------------------------------------
    // parser()
    parser()
    {
        if(1 != yaml_parser_initialize(&parser_))
        {
            throw std::runtime_error(std::string("yaml_parser_initialize() error"));
        }
    }
    //------------------------------------------------------------------
    // ~parser()
    virtual ~parser()
    {
        yaml_parser_delete(&parser_);
    }
    //------------------------------------------------------------------
    // next_document()
    document next_document()
    {
        std::unique_ptr<yaml_document_t> doc_ptr(new yaml_document_t);
        if(1 != yaml_parser_load(parser_addr(), doc_ptr.get()))
        {
            throw std::runtime_error(std::string("yaml_parser_load() error"));
        }
        return document(doc_ptr.release());
    }
    //------------------------------------------------------------------
    parser(const parser&) = delete;
    parser& operator=(const parser&) = delete;

protected:
    yaml_parser_t* parser_addr() { return &parser_; }

private:
    yaml_parser_t parser_;
};

////////////////////////////////////////////////////////////////////////
// yaml::file_parser
class file_parser : public parser {
public:
    //------------------------------------------------------------------
    // file_parser()
    file_parser(const char* fname) :
        fp_(fopen(fname, "r"))
    {
        if(!fp_)
        {
            throw std::system_error(errno, std::generic_category(), "yaml file_parser error");
        }
        yaml_parser_set_input_file(parser_addr(), fp_);
    }
    //------------------------------------------------------------------
    // ~file_parser()
    ~file_parser()
    {
        fclose(fp_);
    }
    //------------------------------------------------------------------
    file_parser(const file_parser&) = delete;
    file_parser& operator=(const file_parser&) = delete;

private:
    FILE* fp_;
};

} // namespace yaml

#endif // !defined(YAML_HPP_INCLUDED_)
