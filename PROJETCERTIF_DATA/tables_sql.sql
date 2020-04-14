CREATE DATABASE retail;
use retail;


-- creation de la table Calendrier
create table calendrier(
    date_calendrier date
)

-- creation de la table clients
create table clients(
    id_client int not null,
    num_compte_client bigint,
    prenom_client varchar(50),
    nom_client varchar(50),
    adresse_client varchar(50),
    ville_client varchar(25),
    province_client varchar(25),
    code_postal_client int(5),
    pays_client varchar(25),
    date_naissance date,
    situation_maritale char(1),
    revenu_annuel varchar(15),
    genre char(1),
    nb_total_enfants int,
    nb_enfants_charge int,
    education varchar(30),
    date_ouverture_compte date,
    carte_fidelite varchar(10),
    profession varchar(15),
    proprietaire char(1)
)