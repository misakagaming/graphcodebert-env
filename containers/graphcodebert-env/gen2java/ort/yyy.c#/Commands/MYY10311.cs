namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: MYY10311_TYPE_CREATE             Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:42
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class MYY10311 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10311_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10311_OA WOa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // LOCAL VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10311_LA WLa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.CYYY0311_IA Cyyy0311Ia;
    GEN.ORT.YYY.CYYY0311_OA Cyyy0311Oa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020145_esc_flag;
    //       +->   MYY10311_TYPE_CREATE              01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Work View imp_reference iyy1_server_data (Transient,
    //       !                     Mandatory, Import only)
    //       !           userid
    //       !           reference_id
    //       !         Entity View imp iyy1_type (Transient, Mandatory, Import
    //       !                     only)
    //       !           tkey_attr_text
    //       !           tsearch_attr_text
    //       !           tother_attr_text
    //       !           tother_attr_date
    //       !           tother_attr_time
    //       !           tother_attr_amount
    //       !       EXPORTS:
    //       !         Entity View exp iyy1_type (Transient, Export only)
    //       !           tinstance_id
    //       !           treference_id
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !       LOCALS:
    //       !         Entity View loc_imp type
    //       !           tkey_attr_text
    //       !           tsearch_attr_text
    //       !           tother_attr_text
    //       !           tother_attr_date
    //       !           tother_attr_time
    //       !           tother_attr_amount
    //       !         Entity View loc_exp type
    //       !           treference_id
    //       !           tinstance_id
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  See the description for the purpose
    //     1 !  
    //     2 !  NOTE: 
    //     2 !  RELEASE HISTORY
    //     2 !  01_00 23-02-1998 New release
    //     2 !  
    //     3 !  NOTE: 
    //     3 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     3 !  !!!!!!!!!!!!
    //     3 !  SET <loc imp*> TO <imp*>
    //     3 !  
    //     4 !  SET loc_imp type tkey_attr_text TO imp iyy1_type
    //     4 !              tkey_attr_text 
    //     5 !  SET loc_imp type tsearch_attr_text TO imp iyy1_type
    //     5 !              tsearch_attr_text 
    //     6 !  SET loc_imp type tother_attr_text TO imp iyy1_type
    //     6 !              tother_attr_text 
    //     7 !  SET loc_imp type tother_attr_date TO imp iyy1_type
    //     7 !              tother_attr_date 
    //     8 !  SET loc_imp type tother_attr_time TO imp iyy1_type
    //     8 !              tother_attr_time 
    //     9 !  SET loc_imp type tother_attr_amount TO imp iyy1_type
    //     9 !              tother_attr_amount 
    //    10 !   
    //    11 !  NOTE: 
    //    11 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    11 !  !!!!!!!!!!!!
    //    11 !  USE <implementation ab>
    //    11 !  
    //    12 !  USE cyyy0311_type_create
    //    12 !     WHICH IMPORTS: Entity View loc_imp type TO Entity View imp
    //    12 !              type
    //    12 !                    Work View imp_reference iyy1_server_data TO
    //    12 !              Work View imp_reference iyy1_server_data
    //    12 !     WHICH EXPORTS: Entity View loc_exp type FROM Entity View
    //    12 !              exp type
    //    12 !                    Work View exp_error iyy1_component FROM Work
    //    12 !              View exp_error iyy1_component
    //    13 !   
    //    14 !  NOTE: 
    //    14 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    14 !  !!!!!!!!!!!!
    //    14 !  SET <exp*> TO <loc exp*>
    //    14 !  
    //    15 !  SET exp iyy1_type tinstance_id TO loc_exp type tinstance_id 
    //    16 !  SET exp iyy1_type treference_id TO loc_exp type treference_id 
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public MYY10311(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:42";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "MYY10311_TYPE_CREATE";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	MYY10311_IA import_view, 
    	MYY10311_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      f_22020145_localAlloc( "MYY10311_TYPE_CREATE" );
      if ( Globdata.GetErrorData().GetLastStatus() == ErrorData.LastStatusIefAllocationError )
      	return;
      
      ++(NestingLevel);
      try {
        f_22020145_init(  );
        f_22020145(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020145(  )
    {
      func_0022020145_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020145" );
      Globdata.GetStateData().SetCurrentABName( "MYY10311_TYPE_CREATE" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    See the description for the purpose                             
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <loc imp*> TO <imp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000004" );
      WLa.LocImpTypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1TypeTkeyAttrText, 4);
      Globdata.GetStateData().SetLastStatementNumber( "0000000005" );
      WLa.LocImpTypeTsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1TypeTsearchAttrText, 20);
      Globdata.GetStateData().SetLastStatementNumber( "0000000006" );
      WLa.LocImpTypeTotherAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1TypeTotherAttrText, 2);
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      WLa.LocImpTypeTotherAttrDate = DateAttr.ValueOf(WIa.ImpIyy1TypeTotherAttrDate);
      Globdata.GetStateData().SetLastStatementNumber( "0000000008" );
      WLa.LocImpTypeTotherAttrTime = TimeAttr.ValueOf(WIa.ImpIyy1TypeTotherAttrTime);
      Globdata.GetStateData().SetLastStatementNumber( "0000000009" );
      WLa.LocImpTypeTotherAttrAmount = DecimalAttr.ValueOf(TIRBDTRU.TruncateToDecimal( WIa.ImpIyy1TypeTotherAttrAmount, 2));
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    USE <implementation ab>                                         
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
      
      Cyyy0311Ia = (GEN.ORT.YYY.CYYY0311_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0311).Assembly,
      	"GEN.ORT.YYY.CYYY0311_IA" ));
      Cyyy0311Oa = (GEN.ORT.YYY.CYYY0311_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0311).Assembly,
      	"GEN.ORT.YYY.CYYY0311_OA" ));
      Cyyy0311Ia.ImpReferenceIyy1ServerDataUserid = FixedStringAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataUserid, 8);
      Cyyy0311Ia.ImpReferenceIyy1ServerDataReferenceId = TimestampAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataReferenceId);
      Cyyy0311Ia.ImpTypeTkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpTypeTkeyAttrText, 4);
      Cyyy0311Ia.ImpTypeTsearchAttrText = FixedStringAttr.ValueOf(WLa.LocImpTypeTsearchAttrText, 20);
      Cyyy0311Ia.ImpTypeTotherAttrText = FixedStringAttr.ValueOf(WLa.LocImpTypeTotherAttrText, 2);
      Cyyy0311Ia.ImpTypeTotherAttrDate = DateAttr.ValueOf(WLa.LocImpTypeTotherAttrDate);
      Cyyy0311Ia.ImpTypeTotherAttrTime = TimeAttr.ValueOf(WLa.LocImpTypeTotherAttrTime);
      Cyyy0311Ia.ImpTypeTotherAttrAmount = DecimalAttr.ValueOf(WLa.LocImpTypeTotherAttrAmount);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY0311).Assembly,
      	"GEN.ORT.YYY.CYYY0311",
      	"Execute",
      	Cyyy0311Ia,
      	Cyyy0311Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020145" );
      Globdata.GetStateData().SetCurrentABName( "MYY10311_TYPE_CREATE" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000012" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy0311Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy0311Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy0311Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy0311Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy0311Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy0311Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy0311Oa.ExpErrorIyy1ComponentChecksum, 15);
      WLa.LocExpTypeTinstanceId = TimestampAttr.ValueOf(Cyyy0311Oa.ExpTypeTinstanceId);
      WLa.LocExpTypeTreferenceId = TimestampAttr.ValueOf(Cyyy0311Oa.ExpTypeTreferenceId);
      Cyyy0311Ia.FreeInstance(  );
      Cyyy0311Ia = null;
      Cyyy0311Oa.FreeInstance(  );
      Cyyy0311Oa = null;
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <exp*> TO <loc exp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000015" );
      WOa.ExpIyy1TypeTinstanceId = TimestampAttr.ValueOf(WLa.LocExpTypeTinstanceId);
      Globdata.GetStateData().SetLastStatementNumber( "0000000016" );
      WOa.ExpIyy1TypeTreferenceId = TimestampAttr.ValueOf(WLa.LocExpTypeTreferenceId);
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020145_localAlloc( String abname )
    {
      // Request localview allocation 
      WLa = (GEN.ORT.YYY.MYY10311_LA)(IefRuntimeParm2.GetInstance( GetType().Assembly,
      	"GEN.ORT.YYY.MYY10311_LA" ));
      if ( WLa == null )
      {
        Globdata.GetStateData().SetCurrentABId( "0022020145" );
        Globdata.GetStateData().SetCurrentABName( abname );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusIefAllocationError );
      }
    }
    
    public void f_22020145_init(  )
    {
      if ( NestingLevel < 2 )
      {
        WLa.Reset();
      }
      WOa.ExpIyy1TypeTinstanceId = "00000000000000000000";
      WOa.ExpIyy1TypeTreferenceId = "00000000000000000000";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
    }
  }// end class
  
}

